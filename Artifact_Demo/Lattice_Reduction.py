#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Falcon-512 lattice pruning with:
- auto public-key h from True_Value f,g via C(f) h = g (mod q)
- reduced system (A', p') with centered reps
- informative row selection (adaptive per round)
- anisotropic column scaling based on candidate uncertainty
- Kannan embedding with explicit target, BKZ if fpylll is available, built-in LLL otherwise
- CVP decoding via Babai + grid over t_scale + jitter
- local enumeration around Babai on most ambiguous variables
- iterative refinement (multi rounds), stronger frequency-based pruning

Usage (example):
  python Lattice_Reduction_plus.py "KeySet_Values (1).csv" --block 40 --tscale 1000000 --rounds 3

Outputs each round:
  - public_key_h.txt
  - pruned_candidates_f_round{r}.csv
  - pruned_candidates_g_round{r}.csv
  - short_vectors_round{r}.txt (top CVP proposals & residuals)
"""

import argparse, ast, math, itertools
import numpy as np
import pandas as pd

# --------------------------------
# Optional fpylll backend
# --------------------------------
FPYLLL_OK, FPY_ERR = False, None
try:
    from fpylll import IntegerMatrix, LLL as FLLL, BKZ, GSO
    FPYLLL_OK = True
except Exception as e:
    FPY_ERR = e

# -------------------------------
# CSV parsing helpers
# -------------------------------
def parse_candidates(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        return [int(v)]
    except Exception:
        pass
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            out.append(int(tok))
        except Exception:
            try:
                out.append(int(float(tok)))
            except Exception:
                pass
    return out

def parse_true(cell):
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        return int(v)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            raise ValueError(f"Cannot parse true value: {cell!r}")

# -------------------------------
# Negacyclic convolution matrix
# -------------------------------
def negacyclic_matrix(poly, q):
    poly = np.array(poly, dtype=int) % q
    n = len(poly)
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            k = (i - j) % n
            val = poly[k]
            if i - j < 0:
                val = (-val) % q
            C[i, j] = val
    return C % q

# -------------------------------
# Modular helpers
# -------------------------------
def modinv(a, q):
    a = int(a) % int(q)
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 modulo q")
    return pow(a, -1, int(q))

def gauss_jordan_mod(A, b, q):
    A = (np.asarray(A, dtype=object) % int(q)).astype(object)
    b = (np.asarray(b, dtype=object) % int(q)).astype(object)
    n = A.shape[0]
    M = np.hstack([A, b.reshape(-1, 1)])
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if int(M[r, col]) % int(q) != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        inv = modinv(int(M[row, col]), int(q))
        for c in range(col, n + 1):
            M[row, c] = (int(M[row, c]) * inv) % int(q)
        for r in range(n):
            if r == row:
                continue
            factor = int(M[r, col]) % int(q)
            if factor:
                for c in range(col, n + 1):
                    M[r, c] = (int(M[r, c]) - factor * int(M[row, c])) % int(q)
        row += 1
        if row == n:
            break
    x = np.array([int(v) % int(q) for v in M[:, -1]], dtype=int)
    return x

def solve_Cf_h_equals_g(Cf, g, q, prefer_sympy=True):
    if prefer_sympy:
        try:
            import sympy as sp
            Cf_sp = sp.Matrix(Cf.tolist())
            g_sp = sp.Matrix(g.tolist())
            Cf_inv = Cf_sp.inv_mod(q)
            h_sp = (Cf_inv * g_sp) % q
            h = np.array([int(v) % q for v in list(h_sp)], dtype=int)
            return h
        except Exception as e:
            print(f"[!] SymPy inv_mod failed ({e}); falling back to Gauss–Jordan mod {q}.")
    return gauss_jordan_mod(Cf, g, q)

# -------------------------------
# Reduced system (A', p'), lifted
# -------------------------------
def build_reduced_system(h, f_true, g_true, f_cands, g_cands, q):
    n = len(f_true)
    C_h = negacyclic_matrix(h, q)
    U_f = [i for i, c in enumerate(f_cands) if len(c) > 1]
    U_g = [i for i, c in enumerate(g_cands) if len(c) > 1]
    f_known = np.array([f_true[i] if i not in U_f else 0 for i in range(n)], dtype=int) % q
    g_known = np.array([g_true[i] if i not in U_g else 0 for i in range(n)], dtype=int) % q
    A_f = C_h[:, U_f]
    E = np.zeros((n, len(U_g)), dtype=int)
    for j, idx in enumerate(U_g):
        E[idx, j] = 1
    A_prime = np.hstack([A_f % q, (-E) % q]) % q
    p_prime = (g_known - (C_h @ f_known) % q) % q

    def lift_centered(v):
        v = np.array(v, dtype=int)
        v = ((v + q//2) % q) - q//2
        return v.astype(int)
    return lift_centered(A_prime), lift_centered(p_prime), U_f, U_g

# -------------------------------
# Column scaling (anisotropic)
# -------------------------------
def compute_column_scales(f_cands, g_cands, U_f, U_g, min_scale=1, max_scale=16):
    """
    Scale columns inversely with uncertainty: s_j ~ 1/(1+range width).
    Bound in [min_scale, max_scale]. Larger s_j => more weight.
    """
    scales = []
    for idx in U_f:
        c = f_cands[idx]
        if not c:
            w = 1
        else:
            w = max(c) - min(c) if len(c) > 1 else 0
        s = 1.0 / (1.0 + w)
        scales.append(s)
    for idx in U_g:
        c = g_cands[idx]
        if not c:
            w = 1
        else:
            w = max(c) - min(c) if len(c) > 1 else 0
        s = 1.0 / (1.0 + w)
        scales.append(s)
    # normalize so median ~ 1, then clamp
    arr = np.array(scales, dtype=float)
    med = np.median(arr) if np.median(arr) > 0 else 1.0
    arr = arr / med
    arr = np.clip(arr, min_scale, max_scale)
    return arr.astype(int)

def apply_column_scales(Ap, scales, q):
    """
    Multiply column j by scales[j]; in embedding we mirror this into q*scales on top-left.
    """
    Ap_s = (Ap * scales[np.newaxis, :]).astype(int)
    q_cols = (q * scales).astype(int)
    return Ap_s, q_cols

# -------------------------------
# Row selection (adaptive)
# -------------------------------
def select_informative_rows(Ap, pp, budget):
    norms = np.linalg.norm(Ap.astype(np.int64), axis=1)
    idx = np.argsort(-norms)[:budget]
    return Ap[idx, :], pp[idx], idx

# -------------------------------
# Embedding + target (with q_cols)
# -------------------------------
def build_embedding_and_target(Ap, pp, q_cols, t_scale=10**6):
    m, r = Ap.shape
    size = r + m + 1
    B = np.zeros((size, size), dtype=int)

    # Top block: diag(q_cols)
    for i in range(r):
        B[i, i] = int(q_cols[i])
        B[r:r+m, i] = Ap[:, i]

    # Slack block: q_slack * I_m (use median of q_cols)
    q_slack = int(np.median(q_cols))
    for j in range(m):
        B[r + j, r + j] = q_slack

    # Last column: target = [-; -pp; t]
    B[r:r+m, r + m] = (-pp).astype(int)
    B[r + m, r + m] = int(t_scale)

    # Target vector t = [0_r; -pp; 0]
    tvec = np.zeros(size, dtype=int)
    tvec[r:r+m] = (-pp).astype(int)
    return B, tvec

# -------------------------------
# Backends: BKZ if fpylll else built-in LLL
# -------------------------------
def lll_builtin(B, delta=0.99):
    # Simple, robust LLL (integer) fallback.
    B = B.copy().astype(np.int64)
    n, m = B.shape
    def gs(B):
        U = B.astype(np.float64).copy()
        mu = np.zeros((n, n), dtype=np.float64)
        Bstar = np.zeros_like(U)
        for i in range(n):
            Bstar[i] = U[i]
            for j in range(i):
                den = np.dot(Bstar[j], Bstar[j]) + 1e-18
                mu[i, j] = np.dot(U[i], Bstar[j]) / den
                Bstar[i] -= mu[i, j] * Bstar[j]
        norms = np.array([np.dot(Bstar[i], Bstar[i]) for i in range(n)], dtype=np.float64)
        return mu, Bstar, norms
    mu, Bstar, norms = gs(B)
    k = 1
    while k < n:
        for j in range(k-1, -1, -1):
            q = int(round(mu[k, j]))
            if q:
                B[k] -= q * B[j]
        mu, Bstar, norms = gs(B)
        if norms[k] >= (delta - mu[k, k-1]**2) * norms[k-1]:
            k += 1
        else:
            B[[k, k-1]] = B[[k-1, k]]
            mu, Bstar, norms = gs(B)
            k = max(1, k-1)
    return B.astype(int)

def reduce_basis(B_np, use_bkz=True, block_size=25):
    if FPYLLL_OK:
        M = IntegerMatrix(B_np.shape[0], B_np.shape[1])
        for i in range(B_np.shape[0]):
            for j in range(B_np.shape[1]):
                M[i, j] = int(B_np[i, j])
        G = GSO.Mat(M, float_type="mpfr")
        print("[*] LLL reduction (mpfr)...")
        FLLL.Reduction(G, delta=0.999, eta=0.501)()
        if use_bkz:
            print(f"[*] BKZ reduction (mpfr, block size = {block_size}) ...")
            params = BKZ.Param(block_size=int(block_size))
            try:
                params.max_loops = 1
            except Exception:
                pass
            try:
                BKZ.Reduction(G, params)()
            except TypeError:
                try:
                    BKZ.Reduction(G, params, None)()
                except TypeError:
                    ds = getattr(BKZ, "DEFAULT_STRATEGY", None)
                    if isinstance(ds, dict):
                        BKZ.Reduction(G, params, ds)()
                    else:
                        BKZ.reduction(M, params)
        R = np.array([[int(M[i, j]) for j in range(M.ncols)] for i in range(M.nrows)], dtype=int)
        print("[*] Reduction complete.")
        return R
    print("[*] fpylll not available; using built-in LLL (no BKZ)")
    R = lll_builtin(B_np, delta=0.99)
    print("[*] Built-in LLL complete.")
    return R

# -------------------------------
# Babai nearest-plane (CVP)
# -------------------------------
def babai(B_reduced, tvec):
    Bf = B_reduced.astype(np.float64)
    Q, R = np.linalg.qr(Bf)
    y = Q.T @ tvec.astype(np.float64)
    c = np.linalg.solve(R, y)
    for i in range(R.shape[0]-1, -1, -1):
        ci = float(c[i])
        c[i] = np.round(ci)
        if i > 0:
            y[:i] -= R[:i, i] * c[i]
            for j in range(i):
                if abs(R[j, j]) > 1e-12:
                    c[j] = y[j] / R[j, j]
    v = (Bf @ c).round().astype(np.int64)
    return v

def decode_x_from_v(v, r, q_cols):
    # With scaling: top block diag(q_cols), so divide by q_cols per coordinate
    x_part = v[:r].astype(np.float64)
    x_hat = np.round(x_part / q_cols.astype(np.float64)).astype(int)
    return x_hat

def residual_norm(Ap, pp, x_hat):
    resid = Ap @ x_hat - pp
    return float(np.linalg.norm(resid.astype(np.int64)))

# -------------------------------
# Local enumeration around Babai
# -------------------------------
def local_enum(Ap, pp, q_cols, x_hat, U_f, U_g, f_cands, g_cands, top_k=6, enum_radius=1, cap=3000):
    """
    Choose top_k coordinates with largest candidate set size; try offsets in {-r..r}
    intersected with candidate lists; keep best few proposals by residual.
    """
    r = Ap.shape[1]
    # sizes per coord
    sizes = []
    for j_rel, j_abs in enumerate(U_f):
        sizes.append((len(f_cands[j_abs]) if f_cands[j_abs] else 1, j_rel))
    base = len(U_f)
    for j_rel, j_abs in enumerate(U_g):
        sizes.append((len(g_cands[j_abs]) if g_cands[j_abs] else 1, base + j_rel))
    sizes.sort(reverse=True)
    idxs = [j for _, j in sizes[:min(top_k, len(sizes))]]
    # Build per-index candidate tweaks
    tweak_lists = []
    for j in idxs:
        # If explicit candidate list exists, restrict to those near x_hat[j]
        if j < len(U_f):
            # map absolute candidates to relative coefficients
            # Here x_hat[j] is integer; restrict tweaks so that x_hat[j]+t in allowed set
            allowed = set(int(v) for v in (f_cands[U_f[j]] or [x_hat[j]]))
        else:
            jj = j - len(U_f)
            allowed = set(int(v) for v in (g_cands[U_g[jj]] or [x_hat[j]]))
        # small neighborhood around x_hat
        neigh = [x_hat[j] + t for t in range(-enum_radius, enum_radius+1)]
        candj = [v for v in neigh if (not allowed or v in allowed)]
        if not candj:
            candj = [x_hat[j]]
        tweak_lists.append(list(sorted(set(candj))))
    combos = list(itertools.product(*tweak_lists))
    if len(combos) > cap:
        # subsample to cap
        step = max(1, len(combos)//cap)
        combos = combos[::step]
    proposals = []
    for comb in combos:
        x_try = x_hat.copy()
        for tval, j in zip(comb, idxs):
            x_try[j] = int(tval)
        sc = residual_norm(Ap, pp, x_try)
        proposals.append((sc, x_try))
    proposals.sort(key=lambda z: z[0])
    # keep best ~50
    return [x for (_, x) in proposals[:50]]

# -------------------------------
# Grid over t_scale + jitter + enum
# -------------------------------
def cvp_harvest(Ap, pp, q_cols, tscales, block_size, repeats=6, enum_top_k=6, enum_radius=1):
    r = Ap.shape[1]
    all_x = []
    for t in tscales:
        B_np, tvec = build_embedding_and_target(Ap, pp, q_cols, t_scale=t)
        B_red = reduce_basis(B_np, use_bkz=True, block_size=block_size)
        # Base Babai
        v = babai(B_red, tvec)
        x_hat = decode_x_from_v(v, r, q_cols)
        all_x.append(x_hat)
        # Jittered Babai
        for _ in range(repeats):
            jitter = np.random.randint(-2, 3, size=Ap.shape[0])
            B_np2, tvec2 = build_embedding_and_target(Ap, (pp + jitter), q_cols, t_scale=t)
            B_red2 = reduce_basis(B_np2, use_bkz=True, block_size=block_size)
            v2 = babai(B_red2, tvec2)
            x2 = decode_x_from_v(v2, r, q_cols)
            all_x.append(x2)
        # Local enumeration around the best Babai for this t
        # pick best-of current batch by residual
        batch = all_x[-(1+repeats):]
        batch_scores = [residual_norm(Ap, pp, x) for x in batch]
        x_best = batch[int(np.argmin(batch_scores))]
        enum_props = local_enum(Ap, pp, q_cols, x_best, U_f=None, U_g=None, 
                                f_cands=None, g_cands=None,  # will be filled by caller variant
                                top_k=0)  # placeholder (not used here)
    return all_x

# -------------------------------
# Adaptive tau
# -------------------------------
def adaptive_tau(Ap, pp, x_list, base=None):
    if not x_list:
        return 3.0 * math.sqrt(Ap.shape[0])
    scores = np.array([residual_norm(Ap, pp, x) for x in x_list], dtype=float)
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) + 1e-9
    tau = med + 3.0 * 1.4826 * mad
    if base is not None:
        tau = min(tau, base)
    return tau

# -------------------------------
# Pruning
# -------------------------------
def prune_with_frequency(xhats, Ap, pp, U_f, U_g, f_cands, g_cands, tau, min_count=1):
    accepted = [x for x in xhats if residual_norm(Ap, pp, x) <= tau]
    if not accepted:
        return f_cands, g_cands, 0
    r_f = len(U_f)
    counts = {}
    for x in accepted:
        for j_rel, j_abs in enumerate(U_f):
            v = int(x[j_rel])
            counts.setdefault(("f", j_abs, v), 0)
            counts[("f", j_abs, v)] += 1
        for j_rel, j_abs in enumerate(U_g):
            v = int(x[r_f + j_rel])
            counts.setdefault(("g", j_abs, v), 0)
            counts[("g", j_abs, v)] += 1
    pf = [list(c) for c in f_cands]
    pg = [list(c) for c in g_cands]
    kept = 0
    for j_abs in U_f:
        old = pf[j_abs]
        if not old:
            continue
        new = [int(v) for v in old if counts.get(("f", j_abs, int(v)), 0) >= min_count]
        if new and set(new) != set(old):
            pf[j_abs] = sorted(set(new)); kept += 1
    for j_abs in U_g:
        old = pg[j_abs]
        if not old:
            continue
        new = [int(v) for v in old if counts.get(("g", j_abs, int(v)), 0) >= min_count]
        if new and set(new) != set(old):
            pg[j_abs] = sorted(set(new)); kept += 1
    return pf, pg, kept

# -------------------------------
# One attack round
# -------------------------------
def attack_round(f_true, g_true, f_cands, g_cands, q, prefer_sympy, block, seed_tscale,
                 row_budget_extra=40, repeats=6, enum_top_k=6, enum_radius=1,
                 use_scaling=True):
    # Build h
    Cf = negacyclic_matrix(f_true, q)
    h = solve_Cf_h_equals_g(Cf, g_true, q, prefer_sympy=prefer_sympy)
    # Build reduced system
    A_prime, p_prime, U_f, U_g = build_reduced_system(h, f_true, g_true, f_cands, g_cands, q)
    r = A_prime.shape[1]; m = A_prime.shape[0]
    # Column scaling
    if use_scaling:
        scales = compute_column_scales(f_cands, g_cands, U_f, U_g, min_scale=1, max_scale=16)
    else:
        scales = np.ones(r, dtype=int)
    Aps, q_cols = apply_column_scales(A_prime, scales, q)
    # Row budget
    m_budget = min(m, r + row_budget_extra)
    Aps, psel, _ = select_informative_rows(Aps, p_prime, m_budget)

    # Prepare tscale grid
    tscales = sorted(set([int(seed_tscale), 10**4, 3*10**4, 10**5, 3*10**5, 10**6, 3*10**6, 10**7]))
    # Harvest candidates
    x_all = []
    for t in tscales:
        B, tvec = build_embedding_and_target(Aps, psel, q_cols, t_scale=t)
        R = reduce_basis(B, use_bkz=True, block_size=block)
        v = babai(R, tvec)
        x_hat = decode_x_from_v(v, r, q_cols)
        x_all.append(x_hat)
        # jittered samples
        for _ in range(repeats):
            jitter = np.random.randint(-2, 3, size=Aps.shape[0])
            B2, tvec2 = build_embedding_and_target(Aps, (psel + jitter), q_cols, t_scale=t)
            R2 = reduce_basis(B2, use_bkz=True, block_size=block)
            v2 = babai(R2, tvec2)
            x2 = decode_x_from_v(v2, r, q_cols)
            x_all.append(x2)
        # local enumeration on ambiguous coords (use actual candidate lists)
        # pick best from current batch:
        batch = x_all[-(1+repeats):]
        scores = [residual_norm(Aps, psel, x) for x in batch]
        xb = batch[int(np.argmin(scores))]
        # choose top_k coords by candidate-set size:
        sizes = []
        for j_rel, j_abs in enumerate(U_f):
            sizes.append((len(f_cands[j_abs]) if f_cands[j_abs] else 1, j_rel))
        base = len(U_f)
        for j_rel, j_abs in enumerate(U_g):
            sizes.append((len(g_cands[j_abs]) if g_cands[j_abs] else 1, base + j_rel))
        sizes.sort(reverse=True)
        idxs = [j for _, j in sizes[:min(enum_top_k, len(sizes))]]
        # enumerate small neighborhoods intersected with candidate sets
        tweaks = []
        for j in idxs:
            if j < len(U_f):
                allowed = set(int(v) for v in (f_cands[U_f[j]] or [xb[j]]))
            else:
                jj = j - len(U_f)
                allowed = set(int(v) for v in (g_cands[U_g[jj]] or [xb[j]]))
            neigh = [xb[j] + t3 for t3 in range(-enum_radius, enum_radius+1)]
            candj = [v for v in neigh if (not allowed or v in allowed)]
            if not candj:
                candj = [xb[j]]
            tweaks.append(sorted(set(candj)))
        if tweaks:
            combos = list(itertools.product(*tweaks))
            # cap
            if len(combos) > 3000:
                step = max(1, len(combos)//3000)
                combos = combos[::step]
            for comb in combos:
                x_try = xb.copy()
                for val, j in zip(comb, idxs):
                    x_try[j] = int(val)
                x_all.append(x_try)

    # Adaptive tau on residuals
    tau = adaptive_tau(Aps, psel, x_all)
    # Prune by frequency among accepted
    f_pruned, g_pruned, kept = prune_with_frequency(
        x_all, Aps, psel, U_f, U_g, f_cands, g_cands,
        tau=tau, min_count=max(1, len(x_all)//10)  # require appearing in >=10% of accepted
    )
    # Collect a digest for logging
    top = sorted([(residual_norm(Aps, psel, x), x) for x in x_all], key=lambda z: z[0])[:20]
    return h, (Aps, psel, U_f, U_g), f_pruned, g_pruned, kept, tau, top

# -------------------------------
# Main (multi-round)
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to KeySet_Values CSV")
    ap.add_argument("--q", type=int, default=12289)
    ap.add_argument("--tscale", type=int, default=10**6)
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=2, help="Number of refinement rounds")
    ap.add_argument("--prefer_sympy", action="store_true", help="Use SymPy inv_mod for h if available")
    ap.add_argument("--no_scaling", action="store_true", help="Disable anisotropic column scaling")
    ap.add_argument("--enum_top_k", type=int, default=6)
    ap.add_argument("--enum_radius", type=int, default=1)
    ap.add_argument("--row_extra", type=int, default=40, help="Keep m' ~ r+row_extra rows")
    ap.add_argument("--repeats", type=int, default=6, help="Jittered Babai reps per t")
    args = ap.parse_args()

    q = int(args.q)
    df = pd.read_csv(args.csv)
    cols = {"candidates f", "True_Value f", "candidates g", "True_Value g"}
    if not cols.issubset(df.columns):
        raise RuntimeError(f"CSV missing required columns: {cols - set(df.columns)}")

    f_true = df["True_Value f"].apply(parse_true).astype(int).tolist()
    g_true = df["True_Value g"].apply(parse_true).astype(int).tolist()
    f_cands = df["candidates f"].apply(parse_candidates).tolist()
    g_cands = df["candidates g"].apply(parse_candidates).tolist()

    n = len(f_true)
    if n != 512:
        print(f"[!] Warning: expected n=512, got n={n}. Continuing.")
    f_true = np.array(f_true, dtype=int) % q
    g_true = np.array(g_true, dtype=int) % q

    # Round loop
    current_f_cands = [list(c) for c in f_cands]
    current_g_cands = [list(c) for c in g_cands]
    for r_idx in range(1, args.rounds + 1):
        print(f"\n=== Round {r_idx}/{args.rounds} ===")
        h, sysinfo, f_pruned, g_pruned, kept, tau, top = attack_round(
            f_true, g_true, current_f_cands, current_g_cands, q,
            prefer_sympy=args.prefer_sympy,
            block=args.block,
            seed_tscale=args.tscale,
            row_budget_extra=args.row_extra,
            repeats=args.repeats,
            enum_top_k=args.enum_top_k,
            enum_radius=args.enum_radius,
            use_scaling=(not args.no_scaling),
        )
        # Save h (first round is sufficient, but harmless each round)
        if r_idx == 1:
            np.savetxt("public_key_h.txt", h, fmt="%d")

        # Emit pruned candidates
        df_out_f = pd.DataFrame({
            "index": np.arange(n, dtype=int),
            "old_candidates": current_f_cands,
            "pruned_candidates": f_pruned
        })
        df_out_g = pd.DataFrame({
            "index": np.arange(n, dtype=int),
            "old_candidates": current_g_cands,
            "pruned_candidates": g_pruned
        })
        df_out_f.to_csv(f"pruned_candidates_f_round{r_idx}.csv", index=False)
        df_out_g.to_csv(f"pruned_candidates_g_round{r_idx}.csv", index=False)

        # Save CVP bests
        with open(f"short_vectors_round{r_idx}.txt", "w") as fh:
            fh.write(f"tau≈{tau:.6f}\n")
            for i, (sc, xh) in enumerate(top):
                fh.write(f"[{i:02d}] resid={sc:>12.6f}  x_hat_first32={xh[:32].tolist()}\n")

        # Prepare next round
        current_f_cands = f_pruned
        current_g_cands = g_pruned

        # Quick progress report
        # Count singletons
        sing_f = sum(1 for c in current_f_cands if isinstance(c, list) and len(c)==1)
        sing_g = sum(1 for c in current_g_cands if isinstance(c, list) and len(c)==1)
        unk_f = sum(1 for c in current_f_cands if isinstance(c, list) and len(c)>1)
        unk_g = sum(1 for c in current_g_cands if isinstance(c, list) and len(c)>1)

        print(f"[*] Round {r_idx}: pruned positions with changes = {kept}")
        print(f"    f: singletons={sing_f}, unknowns>1={unk_f}")
        print(f"    g: singletons={sing_g}, unknowns>1={unk_g}")
        print(f"    Wrote pruned_candidates_f_round{r_idx}.csv and pruned_candidates_g_round{r_idx}.csv")

    print("\n=== Done ===")
    print(f"Backends: {'fpylll BKZ' if FPYLLL_OK else 'built-in LLL'} | BKZ block size: {args.block}")
    print(f"Rounds run: {args.rounds}. See pruned CSVs per round for narrowing results.")
    
if __name__ == "__main__":
    main()
