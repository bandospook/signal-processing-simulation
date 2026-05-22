"""LDPC code: alist parity-check matrix, belief-propagation decoder, encoder."""
from pathlib import Path
import numpy as np
from numba import njit


def _parse_alist(path: str | Path) -> tuple[list[np.ndarray], int, int]:
    """Parse an alist sparse-matrix file into per-check variable-node lists.

    alist layout: line 0 is "n m"; lines 1-3 hold the weight headers; the next
    n lines are the column entries; the final m lines list, per check, the
    1-indexed variable nodes (zero-padded).  Only the row section is needed to
    build the decoder's Tanner graph.
    """
    lines = Path(path).read_text().splitlines()
    n, m = map(int, lines[0].split())
    check_vars = []
    for ln in lines[4 + n: 4 + n + m]:
        vs = [v - 1 for v in map(int, ln.split()) if v != 0]
        check_vars.append(np.array(vs, dtype=np.int64))
    return check_vars, n, m


def _gf2_rref_packed(packed: np.ndarray, m: int, n: int) -> tuple[np.ndarray, int]:
    """In-place GF(2) reduced row echelon form of a bit-packed matrix.

    `packed` is m rows of ceil(n/64) uint64 words.  Returns the pivot column
    index for each of the `rank` pivot rows, and the rank itself.
    """
    one = np.uint64(1)
    pivot_col = np.full(m, -1, dtype=np.int64)
    rank = 0
    for col in range(n):
        word = col >> 6
        shift = np.uint64(col & 63)
        col_bits = (packed[:, word] >> shift) & one
        below = np.nonzero(col_bits[rank:])[0]
        if below.size == 0:
            continue                                  # free (non-pivot) column
        pivot_row = rank + int(below[0])
        if pivot_row != rank:
            packed[[rank, pivot_row]] = packed[[pivot_row, rank]]
            col_bits = (packed[:, word] >> shift) & one
        elim = col_bits.astype(bool)
        elim[rank] = False
        packed[elim] ^= packed[rank]                  # clear the column elsewhere
        pivot_col[rank] = col
        rank += 1
        if rank == m:
            break
    return pivot_col, rank


# _bp_decode runs as Numba-compiled native code, invisible to coverage.py's
# line tracer; it is exercised by the LDPC tests in tests/test_coding.py.
@njit
def _bp_decode(channel_llr, n, m, cn_ptr, edge_vn, vn_ptr, vn_edges, max_iter, alpha):  # pragma: no cover
    """Normalized min-sum belief propagation over the Tanner graph.

    Messages live on edges: msg_v2c (variable->check) and msg_c2v (check->var).
    Each iteration runs a check-node update (sign product times the smallest
    incoming magnitude, excluding the target edge, scaled by alpha), a
    variable-node update, then a parity check on the hard decision for early
    termination.
    """
    n_edges = len(edge_vn)
    msg_v2c = np.empty(n_edges)
    msg_c2v = np.zeros(n_edges)
    total = np.empty(n)
    for e in range(n_edges):
        msg_v2c[e] = channel_llr[edge_vn[e]]

    for _ in range(max_iter):
        # check-node update (normalized min-sum)
        for c in range(m):
            e0 = cn_ptr[c]
            e1 = cn_ptr[c + 1]
            sgn = 1.0
            min1 = 1.0e18
            min2 = 1.0e18
            for e in range(e0, e1):
                x = msg_v2c[e]
                if x < 0.0:
                    sgn = -sgn
                a = abs(x)
                if a < min1:
                    min2 = min1
                    min1 = a
                elif a < min2:
                    min2 = a
            for e in range(e0, e1):
                x = msg_v2c[e]
                mag = min2 if abs(x) == min1 else min1   # exclude the target edge
                s = -sgn if x < 0.0 else sgn              # remove the target's sign
                msg_c2v[e] = alpha * s * mag

        # variable-node update + soft totals
        for v in range(n):
            k0 = vn_ptr[v]
            k1 = vn_ptr[v + 1]
            acc = channel_llr[v]
            for k in range(k0, k1):
                acc += msg_c2v[vn_edges[k]]
            total[v] = acc
            for k in range(k0, k1):
                e = vn_edges[k]
                msg_v2c[e] = acc - msg_c2v[e]

        # parity check on the hard decision -> early termination
        ok = True
        for c in range(m):
            parity = 0
            for e in range(cn_ptr[c], cn_ptr[c + 1]):
                if total[edge_vn[e]] < 0.0:
                    parity ^= 1
            if parity != 0:
                ok = False
                break
        if ok:
            break

    bits = np.empty(n, dtype=np.int64)
    for v in range(n):
        bits[v] = 1 if total[v] < 0.0 else 0
    return bits


class LDPCCode:
    """LDPC code defined by an alist parity-check matrix.

    Loads a sparse parity-check matrix H (n variable nodes, m check nodes),
    decodes soft LLRs with normalized min-sum belief propagation, and encodes
    via a systematic generator derived from H by GF(2) elimination.  The
    generator is built lazily on the first encode() call.
    """

    def __init__(self, alist_path: str | Path) -> None:
        check_vars, n, m = _parse_alist(alist_path)
        self.n = n
        self.m = m
        self.design_rate = (n - m) / n
        cn_deg = np.array([len(cv) for cv in check_vars], dtype=np.int64)
        self._cn_ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(cn_deg)])
        self._edge_vn = np.concatenate(check_vars).astype(np.int64)
        vn_deg = np.bincount(self._edge_vn, minlength=n)
        self._vn_ptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(vn_deg)])
        self._vn_edges = np.argsort(self._edge_vn, kind="stable").astype(np.int64)
        self.info_cols = np.array([], dtype=np.int64)     # set by _build_generator
        self._parity_cols = np.array([], dtype=np.int64)
        self._parity_gen: np.ndarray | None = None
        self.k = 0

    def decode(self, llrs: np.ndarray, max_iter: int = 50, alpha: float = 0.8) -> np.ndarray:
        """Min-sum BP decode of per-bit LLRs; returns the n hard-decision bits."""
        return _bp_decode(np.ascontiguousarray(llrs, dtype=np.float64),
                          self.n, self.m, self._cn_ptr, self._edge_vn,
                          self._vn_ptr, self._vn_edges, max_iter, alpha)

    def build_generator(self) -> None:
        """Derive the systematic generator from H via GF(2) row reduction.

        Idempotent — call it before encode() (encode() also triggers it).  Row-
        reduces H to find rank pivot ("parity") columns; the remaining k =
        n - rank columns carry the message, and each parity bit is the XOR of
        the message bits picked out by its reduced row.  Sets self.k.
        """
        if self._parity_gen is not None:
            return
        n_words = (self.n + 63) // 64
        packed = np.zeros((self.m, n_words), dtype=np.uint64)
        for c in range(self.m):
            for e in range(int(self._cn_ptr[c]), int(self._cn_ptr[c + 1])):
                v = int(self._edge_vn[e])
                packed[c, v >> 6] |= np.uint64(1) << np.uint64(v & 63)

        pivot_col, rank = _gf2_rref_packed(packed, self.m, self.n)
        parity_cols = pivot_col[:rank]
        is_parity = np.zeros(self.n, dtype=bool)
        is_parity[parity_cols] = True
        info_cols = np.nonzero(~is_parity)[0]

        parity_gen = np.zeros((rank, info_cols.size), dtype=np.uint8)
        for j, col in enumerate(info_cols):
            bit = (packed[:rank, col >> 6] >> np.uint64(col & 63)) & np.uint64(1)
            parity_gen[:, j] = bit.astype(np.uint8)

        self.info_cols = info_cols
        self._parity_cols = parity_cols
        self._parity_gen = parity_gen
        self.k = int(info_cols.size)

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """Encode k data bits to an n-bit codeword (builds the generator if needed)."""
        self.build_generator()
        assert self._parity_gen is not None
        message = np.asarray(data_bits, dtype=np.int64)
        codeword = np.zeros(self.n, dtype=np.int64)
        codeword[self.info_cols] = message
        codeword[self._parity_cols] = (self._parity_gen @ message) & 1
        return codeword
