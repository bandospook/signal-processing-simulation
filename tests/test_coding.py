"""Tests for the FEC coding subsystem (sim/coding)."""
from pathlib import Path

import numpy as np
import pytest

from sim.coding import (ConcatenatedCode, ConvolutionalCode, LDPCCode, TurboCode,
                        build_code, decode_frames, encode_frames)
from sim.receiver import soft_demap

_LDPC_ALIST = Path(__file__).resolve().parent.parent / "data" / "ldpc" / "mackay_13298.alist"


# ── Convolutional code ───────────────────────────────────────────────────────

def test_conv_encode_length():
    """Encoded length = (n_data + K - 1) * n  (tail-terminated, rate 1/n)."""
    code = ConvolutionalCode()
    coded = code.encode(np.zeros(100, dtype=int))
    assert len(coded) == (100 + code.K - 1) * code.n


def test_conv_roundtrip_noiseless():
    """Noiseless high-confidence LLRs decode back to the original data exactly."""
    rng = np.random.default_rng(2)
    code = ConvolutionalCode()
    data = rng.integers(0, 2, 500)
    coded = code.encode(data)
    llrs = np.where(coded == 0, 30.0, -30.0)
    decoded = code.decode(llrs)
    assert np.array_equal(decoded, data)


def test_conv_coding_gain():
    """Soft-decision Viterbi beats uncoded BPSK at the same information Eb/N0."""
    rng = np.random.default_rng(0)
    code = ConvolutionalCode()
    n_data = 5000
    data = rng.integers(0, 2, n_data)
    ebn0 = 10.0 ** (5.0 / 10.0)

    # Coded path — rate 1/2, so each coded symbol carries half the info-bit energy.
    coded = code.encode(data)
    tx = 1.0 - 2.0 * coded
    esn0 = ebn0 * code.rate
    sigma = np.sqrt(1.0 / (2.0 * esn0))
    rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
    coded_ber = float(np.mean(code.decode(llrs) != data))

    # Uncoded BPSK at the same information Eb/N0.
    sigma_u = np.sqrt(1.0 / (2.0 * ebn0))
    rx_u = (1.0 - 2.0 * data) + sigma_u * rng.standard_normal(n_data)
    uncoded_ber = float(np.mean((rx_u < 0) != data))

    assert coded_ber < uncoded_ber, f"no coding gain: {coded_ber} vs {uncoded_ber}"
    assert coded_ber < 0.01, f"coded BER too high: {coded_ber}"


def test_conv_weight_spectrum():
    """The (171,133) code has free distance 10 and the textbook weight spectrum."""
    spectrum = ConvolutionalCode().weight_spectrum(18)
    assert int(np.argmax(spectrum > 0)) == 10            # free distance
    assert np.all(spectrum[:10] == 0)
    assert spectrum[10] == 36                            # B at d_free (textbook)
    assert spectrum[12] == 211 and spectrum[14] == 1404


def test_conv_union_bound_monotone():
    """The Viterbi union bound is a positive, monotonically decreasing BER curve."""
    code = ConvolutionalCode()
    vals = [code.union_bound_ber(e) for e in (2.0, 4.0, 6.0, 8.0)]
    assert all(0.0 < v < 1.0 for v in vals)
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


# ── LDPC code ────────────────────────────────────────────────────────────────

def test_ldpc_parses():
    """The alist parity-check matrix loads with the expected dimensions."""
    code = LDPCCode(_LDPC_ALIST)
    assert code.n == 13298
    assert code.m == 10002
    assert 0.2 < code.design_rate < 0.3


def test_ldpc_decodes_clean():
    """A near-noiseless all-zero codeword decodes to all zeros."""
    code = LDPCCode(_LDPC_ALIST)
    rng = np.random.default_rng(0)
    sigma = 0.05
    rx = 1.0 + sigma * (rng.standard_normal(code.n) + 1j * rng.standard_normal(code.n))
    llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
    assert np.all(code.decode(llrs) == 0)


@pytest.fixture(scope="module")
def ldpc_with_generator():
    """LDPC code with its systematic generator built once, shared across encode tests."""
    code = LDPCCode(_LDPC_ALIST)
    code.build_generator()
    return code


def test_ldpc_encode_valid_codeword(ldpc_with_generator):
    """Every encoded codeword satisfies all parity checks of H."""
    code = ldpc_with_generator
    rng = np.random.default_rng(10)
    codeword = code.encode(rng.integers(0, 2, code.k))
    syndrome = np.bitwise_xor.reduceat(codeword[code._edge_vn], code._cn_ptr[:-1])
    assert np.all(syndrome == 0)


def test_ldpc_encode_decode_roundtrip(ldpc_with_generator):
    """Encode then noiseless decode recovers the message in the info columns."""
    code = ldpc_with_generator
    rng = np.random.default_rng(11)
    data = rng.integers(0, 2, code.k)
    codeword = code.encode(data)
    llrs = np.where(codeword == 0, 30.0, -30.0)
    assert np.array_equal(code.decode(llrs)[code.info_cols], data)


def test_ldpc_coding_gain():
    """All-zero codeword over AWGN: BP decoding beats uncoded BPSK at same Eb/N0.

    The code is linear, so BER is independent of the transmitted codeword; the
    all-zero codeword is the conventional choice for LDPC BER simulation.
    """
    code = LDPCCode(_LDPC_ALIST)
    rng = np.random.default_rng(1)
    ebn0 = 10.0 ** (4.0 / 10.0)
    n_frames = 5

    # Coded path — each coded symbol carries design_rate info-bits of energy.
    esn0 = ebn0 * code.design_rate
    sigma = np.sqrt(1.0 / (2.0 * esn0))
    coded_errs = 0
    for _ in range(n_frames):
        rx = 1.0 + sigma * (rng.standard_normal(code.n) + 1j * rng.standard_normal(code.n))
        llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
        coded_errs += int(np.sum(code.decode(llrs) != 0))
    coded_ber = coded_errs / (n_frames * code.n)

    # Uncoded BPSK at the same information Eb/N0.
    sigma_u = np.sqrt(1.0 / (2.0 * ebn0))
    rx_u = 1.0 + sigma_u * rng.standard_normal(n_frames * code.n)
    uncoded_ber = float(np.mean(rx_u < 0))

    assert coded_ber < uncoded_ber, f"no coding gain: {coded_ber} vs {uncoded_ber}"
    assert coded_ber < 0.005, f"coded BER too high: {coded_ber}"


# ── Turbo code ───────────────────────────────────────────────────────────────

def test_turbo_encode_length():
    """Rate-1/3 turbo: encoded length is 3 * k (systematic + two parity streams)."""
    code = TurboCode(500)
    assert len(code.encode(np.zeros(500, dtype=int))) == 3 * 500


def test_turbo_roundtrip_noiseless():
    """Noiseless high-confidence LLRs decode back to the original data exactly."""
    rng = np.random.default_rng(4)
    code = TurboCode(800)
    data = rng.integers(0, 2, 800)
    coded = code.encode(data)
    llrs = np.where(coded == 0, 30.0, -30.0)
    decoded = code.decode(llrs, iterations=4)
    assert np.array_equal(decoded, data)


def test_turbo_coding_gain():
    """Iterative BCJR turbo decoding beats uncoded BPSK at the same Eb/N0."""
    rng = np.random.default_rng(5)
    code = TurboCode(2000)
    data = rng.integers(0, 2, code.k)
    ebn0 = 10.0 ** (2.5 / 10.0)

    coded = code.encode(data)
    tx = 1.0 - 2.0 * coded
    esn0 = ebn0 * code.rate
    sigma = np.sqrt(1.0 / (2.0 * esn0))
    rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
    coded_ber = float(np.mean(code.decode(llrs, iterations=8) != data))

    sigma_u = np.sqrt(1.0 / (2.0 * ebn0))
    rx_u = (1.0 - 2.0 * data) + sigma_u * rng.standard_normal(code.k)
    uncoded_ber = float(np.mean((rx_u < 0) != data))

    assert coded_ber < uncoded_ber, f"no coding gain: {coded_ber} vs {uncoded_ber}"
    assert coded_ber < 0.01, f"coded BER too high: {coded_ber}"


# ── Concatenated code (Reed-Solomon outer + convolutional inner) ─────────────

def test_concat_encode_length():
    """Encoded length matches the reported coded_bits."""
    code = ConcatenatedCode()
    coded = code.encode(np.zeros(code.k, dtype=int))
    assert len(coded) == code.coded_bits


def test_concat_roundtrip_noiseless():
    """Noiseless high-confidence LLRs decode back to the original data exactly."""
    rng = np.random.default_rng(6)
    code = ConcatenatedCode()
    data = rng.integers(0, 2, code.k)
    coded = code.encode(data)
    llrs = np.where(coded == 0, 30.0, -30.0)
    assert np.array_equal(code.decode(llrs), data)


def test_concat_coding_gain():
    """Concatenated (RS + convolutional) decoding beats uncoded BPSK at same Eb/N0."""
    rng = np.random.default_rng(7)
    code = ConcatenatedCode()
    ebn0 = 10.0 ** (4.0 / 10.0)
    n_frames = 3

    sigma = np.sqrt(1.0 / (2.0 * ebn0 * code.rate))
    errs = 0
    for _ in range(n_frames):
        data = rng.integers(0, 2, code.k)
        tx = 1.0 - 2.0 * code.encode(data)
        rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
        llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
        errs += int(np.sum(code.decode(llrs) != data))
    coded_ber = errs / (n_frames * code.k)

    nbits = n_frames * code.k
    data_u = rng.integers(0, 2, nbits)
    sigma_u = np.sqrt(1.0 / (2.0 * ebn0))
    rx_u = (1.0 - 2.0 * data_u) + sigma_u * rng.standard_normal(nbits)
    uncoded_ber = float(np.mean((rx_u < 0) != data_u))

    assert coded_ber < uncoded_ber, f"no coding gain: {coded_ber} vs {uncoded_ber}"
    assert coded_ber < 0.005, f"coded BER too high: {coded_ber}"


# ── Parallel batch decoding ──────────────────────────────────────────────────

def test_decode_batch_matches_decode():
    """decode_batch reproduces looping decode per frame, exactly, for every code."""
    rng = np.random.default_rng(20)

    conv = ConvolutionalCode()
    cf = rng.standard_normal((4, 1000))
    cb = conv.decode_batch(cf)
    assert all(np.array_equal(cb[i], conv.decode(cf[i])) for i in range(len(cf)))

    turbo = TurboCode(300)
    tf = rng.standard_normal((4, 900))
    tb = turbo.decode_batch(tf)
    assert all(np.array_equal(tb[i], turbo.decode(tf[i])) for i in range(len(tf)))

    concat = ConcatenatedCode()
    qf = rng.standard_normal((3, concat.coded_bits))
    qb = concat.decode_batch(qf)
    assert all(np.array_equal(qb[i], concat.decode(qf[i])) for i in range(len(qf)))

    ldpc = LDPCCode(_LDPC_ALIST)
    lf = rng.standard_normal((3, ldpc.n))
    lb = ldpc.decode_batch(lf)
    assert all(np.array_equal(lb[i], ldpc.decode(lf[i])) for i in range(len(lf)))


def test_decode_data_recovers_frames(ldpc_with_generator):
    """Every code exposes .k and round-trips multi-frame data through decode_data."""
    rng = np.random.default_rng(21)
    codes = [ConvolutionalCode(), TurboCode(300), ConcatenatedCode(), ldpc_with_generator]
    for code in codes:
        data = np.stack([rng.integers(0, 2, code.k) for _ in range(3)])
        coded = np.stack([code.encode(row) for row in data])
        llrs = np.where(coded == 0, 30.0, -30.0).astype(float)
        assert np.array_equal(code.decode_data(llrs), data)


# ── Code factory ─────────────────────────────────────────────────────────────

def test_build_code():
    """build_code constructs the right code object for each scheme."""
    assert isinstance(build_code({"scheme": "convolutional"}), ConvolutionalCode)
    assert isinstance(build_code({"scheme": "concatenated"}), ConcatenatedCode)
    assert isinstance(build_code({"scheme": "turbo", "block_length": 400}), TurboCode)
    assert isinstance(build_code({"scheme": "ldpc", "matrix": _LDPC_ALIST}), LDPCCode)


def test_build_code_unknown_scheme():
    """build_code rejects an unknown coding scheme."""
    with pytest.raises(ValueError, match="Unknown coding scheme"):
        build_code({"scheme": "polar"})


def test_build_code_ldpc_default_matrix():
    """ldpc scheme uses the bundled default alist when matrix is missing or blank."""
    from sim.coding import DEFAULT_LDPC_MATRIX
    assert DEFAULT_LDPC_MATRIX.exists()
    assert isinstance(build_code({"scheme": "ldpc"}), LDPCCode)
    assert isinstance(build_code({"scheme": "ldpc", "matrix": ""}), LDPCCode)


def test_encode_decode_frames(ldpc_with_generator):
    """encode_frames / decode_frames round-trip multi-frame data, LDPC and otherwise."""
    rng = np.random.default_rng(30)
    for code in (ConvolutionalCode(block_length=200), ldpc_with_generator):
        data, coded = encode_frames(code, 3, rng)
        assert len(data) == 3 * code.k
        llrs = np.where(coded == 0, 25.0, -25.0).astype(float)
        decoded = decode_frames(code, llrs, 3)
        assert np.array_equal(decoded, data)


def test_decode_frames_pads_short_llrs():
    """decode_frames pads with zeros when LLR count is shorter than n_frames * code.coded_bits."""
    rng = np.random.default_rng(42)
    code = ConvolutionalCode(block_length=200)
    data, coded = encode_frames(code, 2, rng)
    llrs = np.where(coded == 0, 25.0, -25.0).astype(float)
    # Trim a few LLRs to simulate transient stripping — decode_frames should pad and succeed.
    trimmed = llrs[:-6]
    decoded = decode_frames(code, trimmed, 2)
    assert decoded.shape == data.shape
