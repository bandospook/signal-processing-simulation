"""LDPC code: alist parity-check matrix + Numba belief-propagation decoder."""
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
    """LDPC code defined by an alist parity-check matrix, with a min-sum decoder.

    Loads a sparse parity-check matrix H (n variable nodes, m check nodes) and
    decodes soft LLRs with normalized min-sum belief propagation.  The decoder
    needs only H; encoding is handled elsewhere.
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

    def decode(self, llrs: np.ndarray, max_iter: int = 50, alpha: float = 0.8) -> np.ndarray:
        """Min-sum BP decode of per-bit LLRs; returns the n hard-decision bits."""
        return _bp_decode(np.ascontiguousarray(llrs, dtype=np.float64),
                          self.n, self.m, self._cn_ptr, self._edge_vn,
                          self._vn_ptr, self._vn_edges, max_iter, alpha)
