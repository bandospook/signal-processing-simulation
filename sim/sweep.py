from collections.abc import Callable

from .simulation import wideband_bpsk_simulation

_PrintCB = Callable[[str], None] | None


_PointCB = Callable[[int, int], None] | None


def parameter_sweep(carriers: list[dict],
                    sample_rate: float,
                    am_am_cfg: dict,
                    am_pm_cfg: dict,
                    ibo_db_values: list[float],
                    noise_density_dbfs_values: list[float],
                    ola_filter_span: int = 16,
                    ola_block_size: int = 4096,
                    seed: int | None = None,
                    chunk_print: _PrintCB = None,
                    point_cb: _PointCB = None) -> tuple[dict, list[dict]]:
    """
    Run the simulation on a 2-D grid of IBO × noise density values.

    Returns (first_sim, results) where `first_sim` is the full
    wideband_bpsk_simulation return dict for the first grid point (used by the
    caller to draw the wideband PSD plot), and `results` is a list of compact
    per-point dicts:
        ibo_db              float
        noise_density_dbfs  float
        carriers            list of {name, cnr_db, cir_db, cnir_db, evm_rms, ber}
    """
    n_total = len(ibo_db_values) * len(noise_density_dbfs_values)
    n_done  = 0
    results: list[dict] = []
    first_sim: dict | None = None

    # Carriers with sweep_demod=False contribute to the wideband composite but
    # their per-carrier demod (BER/EVM/CNR/CIR/CNIR) is skipped each grid point.
    demod_carriers = {c["name"] for c in carriers if c.get("sweep_demod", False)}

    for ibo in ibo_db_values:
        for noise in noise_density_dbfs_values:
            sim = wideband_bpsk_simulation(
                carriers=carriers,
                sample_rate=sample_rate,
                am_am_cfg=am_am_cfg,
                am_pm_cfg=am_pm_cfg,
                input_backoff_db=ibo,
                noise_density_dbfs=noise,
                ola_filter_span=ola_filter_span,
                ola_block_size=ola_block_size,
                seed=seed,
                demod_carriers=demod_carriers,
                chunk_print=chunk_print,
            )
            if first_sim is None:
                first_sim = sim
            results.append(dict(
                ibo_db=ibo,
                noise_density_dbfs=noise,
                carriers=[
                    dict(
                        name=cr["name"],
                        cnr_db=cr["cnr_db"],
                        cir_db=cr["cir_db"],
                        cnir_db=cr["cnir_db"],
                        evm_rms=cr["evm_rms"],
                        ber=cr["ber"],
                    )
                    for cr in sim["carriers"]
                ],
            ))
            n_done += 1
            if point_cb is not None:
                point_cb(n_done, n_total)
            print(f"  [{n_done:>{len(str(n_total))}}/{n_total}] "
                  f"IBO={ibo:.1f} dB  noise={noise:.1f} dBFS/Hz  done")

    assert first_sim is not None  # n_total >= 1 guaranteed by caller
    return first_sim, results
