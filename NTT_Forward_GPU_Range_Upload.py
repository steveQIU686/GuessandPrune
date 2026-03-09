# ================================================================
# GPU-accelerated forward inference (Windows + RTX 3090 friendly)
# - Vectorizes pair enumeration inside each butterfly with CuPy
# - Optional multi-butterfly parallelism via ThreadPoolExecutor
# - Falls back to CPU if CuPy is unavailable
# ================================================================

import os
import itertools
import pandas as pd
import numpy as np
from sympy import mod_inverse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


# ---- Require CuPy/CUDA (exit if unavailable) ----
try:
    import cupy as cp
    from cupy.cuda.runtime import getDeviceCount
    # Ensure at least one CUDA device is present
    if getDeviceCount() < 1:
        raise RuntimeError("No CUDA-capable device found.")
except Exception as e:
    # Fail fast with a clear error message
    raise SystemExit(
        "ERROR: CuPy/CUDA is required for this program but is not available.\n"
        f"Details: {e}"
    )

# -----------------------------
# Constants and global settings
# -----------------------------
q = 12289
R = 2**16
Rinv = mod_inverse(R, q)

# Use zip for forward CSVs (same as original)
CSV_COMPRESSION = "zip"

# Ensure output folders exist (Windows safe)
def _ensure_dirs():
    os.makedirs("C:/Users/jqiu2/Desktop/Artifact_Demo/Data/GPU/for_Backward", exist_ok=True)
    os.makedirs("C:/Users/jqiu2/Desktop/Artifact_Demo/Data/GPU/forward_data_bkup", exist_ok=True)

_ensure_dirs()

# --------------------------------
# Twiddle (GMb) in Montgomery form
# --------------------------------
GMb_mont = [
    4091,  7888, 11060, 11208,  6960,  4342,  6275,  9759,
    1591,  6399,  9477,  5266,   586,  5825,  7538,  9710,
    1134,  6407,  1711,   965,  7099,  7674,  3743,  6442,
    10414,  8100,  1885,  1688,  1364, 10329, 10164,  9180,
    12210,  6240,   997,   117,  4783,  4407,  1549,  7072,
    2829,  6458,  4431,  8877,  7144,  2564,  5664,  4042,
    12189,   432, 10751,  1237,  7610,  1534,  3983,  7863,
    2181,  6308,  8720,  6570,  4843,  1690,    14,  3872,
    5569,  9368, 12163,  2019,  7543,  2315,  4673,  7340,
    1553,  1156,  8401, 11389,  1020,  2967, 10772,  7045,
    3316, 11236,  5285, 11578, 10637, 10086,  9493,  6180,
    9277,  6130,  3323,   883, 10469,   489,  1502,  2851,
    11061,  9729,  2742, 12241,  4970, 10481, 10078,  1195,
    730,  1762,  3854,  2030,  5892, 10922,  9020,  5274,
    9179,  3604,  3782, 10206,  3180,  3467,  4668,  2446,
    7613,  9386,   834,  7703,  6836,  3403,  5351, 12276,
    3580,  1739, 10820,  9787, 10209,  4070, 12250,  8525,
    10401,  2749,  7338, 10574,  6040,   943,  9330,  1477,
    6865,  9668,  3585,  6633, 12145,  4063,  3684,  7680,
    8188,  6902,  3533,  9807,  6090,   727, 10099,  7003,
    6945,  1949,  9731, 10559,  6057,   378,  7871,  8763,
    8901,  9229,  8846,  4551,  9589, 11664,  7630,  8821,
    5680,  4956,  6251,  8388, 10156,  8723,  2341,  3159,
    1467,  5460,  8553,  7783,  2649,  2320,  9036,  6188,
    737,  3698,  4699,  5753,  9046,  3687,    16,   914,
    5186, 10531,  4552,  1964,  3509,  8436,  7516,  5381,
    10733,  3281,  7037,  1060,  2895,  7156,  8887,  5357,
    6409,  8197,  2962,  6375,  5064,  6634,  5625,   278,
    932, 10229,  8927,  7642,   351,  9298,   237,  5858,
    7692,  3146, 12126,  7586,  2053, 11285,  3802,  5204,
    4602,  1748, 11300,   340,  3711,  4614,   300, 10993,
    5070, 10049, 11616, 12247,  7421, 10707,  5746,  5654,
    3835,  5553,  1224,  8476,  9237,  3845,   250, 11209,
    4225,  6326,  9680, 12254,  4136,  2778,   692,  8808,
    6410,  6718, 10105, 10418,  3759,  7356, 11361,  8433,
    6437,  3652,  6342,  8978,  5391,  2272,  6476,  7416,
    8418, 10824, 11986,  5733,   876,  7030,  2167,  2436,
    3442,  9217,  8206,  4858,  5964,  2746,  7178,  1434,
    7389,  8879, 10661, 11457,  4220,  1432, 10832,  4328,
    8557,  1867,  9454,  2416,  3816,  9076,   686,  5393,
    2523,  4339,  6115,   619,   937,  2834,  7775,  3279,
    2363,  7488,  6112,  5056,   824, 10204, 11690,  1113,
    2727,  9848,   896,  2028,  5075,  2654, 10464,  7884,
    12169,  5434,  3070,  6400,  9132, 11672, 12153,  4520,
    1273,  9739,  11468,  9937, 10039,  9720,  2262,  9399,
    11192,   315,  4511,  1158,  6061,  6751, 11865,   357,
    7367,  4550,   983,  8534,  8352, 10126,  7530,  9253,
    4367,  5221,  3999,  8777,  3161,  6990,  4130,  11652,
    3374, 11477,  1753,   292,  8681,  2806, 10378, 12188,
    5800, 11811,  3181,  1988,  1024,  9340,  2477, 10928,
    4582,  6750,  3619,  5503,  5233,  2463,  8470,  7650,
    7964,  6395,  1071,  1272,  3474, 11045,  3291, 11344,
    8502,  9478,  9837,  1253,  1857,  6233,  4720,  11561,
    6034,  9817,  3339,  1797,  2879,  6242,  5200,  2114,
    7962,  9353,  11363,  5475,  6084,  9601,  4108,  7323,
    10438,  9471,  1271,   408,  6911,  3079,   360,  8276,
    11535,  9156,  9049,  11539,   850,  8617,   784,  7919,
    8334,  12170,  1846, 10213, 12184,  7827, 11903,  5600,
    9779,  1012,   721,  2784,  6676,  6552,  5348,  4424,
    6816,  8405,  9959,  5150,  2356,  5552,  5267,  1333,
    8801,  9661,  7308,  5788,  4910,   909, 11613,  4395,
    8238,  6686,  4302,  3044,  2285, 12249,  1963,  9216,
    4296, 11918,   695,  4371,  9793,  4884,  2411, 10230,
    2650,   841,  3890, 10231,  7248,  8505, 11196,  6688,
    4059,  6060,  3686,  4722, 11853,  5816,  7058,  6868,
    11137,  7926,  4894, 12284,  4102,  3908,  3610,  6525,
    7938,  7982, 11977,  6755,   537,  4562,  1623,  8227,
    11453,  7544,   906, 11816,  9548, 10858,  9703,  2815,
    11736,  6813,  6979,   819,  8903,  6271, 10843,   348,
    7514,  8339,  6439,   694,   852,  5659,  2781,  3716,
    11589,  3024,  1523,  8659,  4114, 10738,  3303,  5885,
    2978,  7289, 11884,  9123,  9323, 11830,    98,  2526,
    2116,  4131, 11407,  1844,  3645,  3916,  8133,  2224,
    10871,  8092,  9651,  5989,  7140,  8480,  1670,   159,
    10923,  4918,   128,  7312,   725,  9157,  5006,  6393,
    3494,  6043, 10972,  6181, 11838,  3423, 10514,  7668,
    3693,  6658,  6905,  11953, 10212, 11922,  9101,  8365,
    5110,    45,  2400,  1921,  4377,  2720,  1695,    51,
    2808,   650,  1896,  9997,  9971,  11980,  8098,  4833,
    4135,  4257,  5838,  4765, 10985, 11532,   590, 12198,
    482, 12173,  2006,  7064, 10018,  3912, 12016, 10519,
    11362,  6954,  2210,   284,  5413,  6601,  3865, 10339,
    11188,  6231,   517,  9564, 11281,  3863,  1210,  4604,
    8160,  11447,   153,  7204,  5763,  5089,  9248, 12154,
    11748,  1354,  6672,   179,  5532,  2646,  5941, 12185,
    862,  3158,   477,  7279,  5678,  7914,  4254,   302,
    2893, 10114,  6890,  9560,  9647, 11905,  4098,  9824,
    10269,  1353, 10715,  5325,  6254,  3951,  1807,  6449,
    5159,  1308,  8315,  3404,  1877,  1231,   112,  6398,
    11724, 12272,  7286,  1459, 12274,  9896,  3456,   800,
    1397, 10678,   103,  7420,  7976,   936,   764,   632,
    7996,  8223,  8445,  7758, 10870,  9571,  2508,  1946,
    6524, 10158,  1044,  4338,  2457,  3641,  1659,  4139,
    4688,  9733, 11148,  3946,  2082,  5261,  2036, 11850,
    7636, 12236,  5366,  2380,  1399,  7720,  2100,  3217,
    10912,  8898,  7578, 11995,  2791,  1215,  3355,  2711,
    2267,  2004,  8568, 10176,  3214,  2337,  1750,  4729,
    4997,  7415,  6315, 12044,  4374,  7157,  4844,   211,
    8003, 10159,  9290, 11481,  1735,  2336,  5793,  9875,
    8192,   986,  7527,  1401,   870,  3615,  8465,  2756,
    9770,  2034, 10168,  3264,  6132,    54,  2880,  4763,
    11805,  3074,  8286,  9428,  4881,  6933,  1090, 10038,
    2567,   708,   893,  6465,  4962, 10024,  2090,  5718,
    10743,   780,  4733,  4623,  2134,  2087,  4802,   884,
    5372,  5795,  5938,  4333,  6559,  7549,  5269, 10664,
    4252,  3260,  5917, 10814,  5768,  9983,  8096,  7791,
    6800,  7491,  6272,  1907, 10947,  6289, 11803,  6032,
    11449,  1171,  9201,  7933,  2479,  7970, 11337,  7062,
    8911,  6728,  6542,  8114,  8828,  6595,  3545,  4348,
    4610,  2205,  6999,  8106,  5560, 10390,  9321,  2499,
    2413,  7272,  6881, 10582,  9308,  9437,  3554,  3326,
    5991,  11969,  3415, 12283,  9838, 12063,  4332,  7830,
    11329,  6605, 12271,  2044, 11611,  7353, 11201, 11582,
    3733,  8943,  9978,  1627,  7168,  3935,  5050,  2762,
    7496, 10383,   755,  1654, 12053,  4952, 10134,  4394,
    6592,  7898,  7497,  8904, 12029,  3581, 10748,  5674,
    10358,  4901,  7414,  8771,   710,  6764,  8462,  7193,
    5371,  7274,  11084,   290,  7864,  6827, 11822,  2509,
    6578,  4026,  5807,  1458,  5721,  5762,  4178,  2105,
    11621,  4852,  8897,  2856, 11510,  9264,  2520,  8776,
    7011,  2647,  1898,  7039,  5950, 11163,  5488,  6277,
    9182,  11456,   633, 10046, 11554,  5633,  9587,  2333,
    7008,  7084,  5047,  7199,  9865,  8997,   569,  6390,
    10845,  9679,  8268, 11472,  4203,  1997,     2,  9331,
    162,  6182,  2000,  3649,  9792,  6363,  7557,  6187,
    8510,  9935,  5536,  9019,  3706, 12009,  1452,  3067,
    5494,  9692,  4865,  6019,  7106,  9610,  4588, 10165,
    6261,  5887,  2652, 10172,  1580, 10379,  4638,  9949
]

CSV_COMPRESSION = "zip"  # matches saved format

def read_forward_csv(path: str) -> pd.DataFrame:
    """Read a zip-compressed CSV."""
    return pd.read_csv(path, compression=CSV_COMPRESSION)

def _fix_windows_drive(path: str) -> str:
    if re.match(r'^[A-Za-z]:[^\\/]', path):
        return path[:2] + os.sep + path[2:]
    return path

def _resolve_file(folder: str, base: str, i: int) -> str:
    candidates = [
        os.path.join(folder, f"{base}{i}"),
        os.path.join(folder, f"{base}{i}.csv"),
        os.path.join(folder, f"{base}{i}.csv.zip"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(folder, f"{base}{i}*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Missing file for butterfly {i}: {candidates}")

def _normalize_colname(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("’", "'").replace("′", "'").replace("`", "'").replace("´", "'")
    s = s.replace("'", "prime")
    return re.sub(r"[^0-9a-z_]+", "", s)

def _resolve_prime_columns(df: pd.DataFrame):
    norm_map = {col: _normalize_colname(col) for col in df.columns}
    col_a1p = next((c for c, n in norm_map.items() if n == "a1prime"), None)
    col_a2p = next((c for c, n in norm_map.items() if n == "a2prime"), None)
    if col_a1p is None or col_a2p is None:
        raise KeyError(f"Could not locate columns a1'/a2'. Columns: {list(df.columns)}")
    return col_a1p, col_a2p

def _unique_sorted_int_series(s: pd.Series):
    """Coerce to int, drop NaN, deduplicate, then sort ascending."""
    s_num = pd.to_numeric(s, errors="coerce").dropna().astype(int)
    return sorted(set(s_num.tolist()))

def build_Aval_67(folder: str, base_name: str = "forward_stage6_butterfly"):
    """
    Build Aval_67 (length 512) from 256 zip-compressed CSVs in `folder`.

    Mapping (file i -> rows):
      g = i // 2, r = i % 2
      a1' -> row (4*g + r)
      a2' -> row (4*g + r + 2)
      => (0,2), (1,3), (4,6), (5,7), ..., (508,510), (509,511)
    """
    folder = _fix_windows_drive(folder)
    folder = os.path.normpath(folder)

    Aval_67 = [[] for _ in range(512)]

    for i in range(256): 
        print("loading stage 6 butterfly" + str(i))
        path = _resolve_file(folder, base_name, i)

        # Read header to resolve column names
        header_df = read_forward_csv(path).head(0)
        col_a1p, col_a2p = _resolve_prime_columns(header_df)

        # Read needed columns
        df = read_forward_csv(path).loc[:, [col_a1p, col_a2p]]

        a1_vals = _unique_sorted_int_series(df[col_a1p])
        a2_vals = _unique_sorted_int_series(df[col_a2p])

        g, r = divmod(i, 2)
        idx_a1 = 4 * g + r
        idx_a2 = idx_a1 + 2

        Aval_67[idx_a1] = a1_vals
        Aval_67[idx_a2] = a2_vals

    return Aval_67

# ----------------------------------------
# Shared: save forward CSVs to both paths
# ----------------------------------------
def _save_forward_csvs(trace_df, stage, bfly_idx):
    filename = f"C:/Users/jqiu2/Desktop/Artifact_Demo/Data/GPU/for_Backward/for_Backward_stage{stage}_butterfly{bfly_idx}.csv"
    write_forward_csv(trace_df, filename)
    filename2 = f"C:/Users/jqiu2/Desktop/Artifact_Demo/Data/GPU/forward_data_bkup/forward_stage{stage}_butterfly{bfly_idx}.csv"
    write_forward_csv(trace_df, filename2) 

# ------------------------------------------------------
# Public butterfly wrapper (chooses GPU or CPU path)
# ------------------------------------------------------
def butterfly_operation_strict(a1_range, a2_range, w, norm_twiddle, norm_add, norm_sub,
                               stage, bfly_idx, q=12289):
    return butterfly_operation_strict_gpu(a1_range, a2_range, w, norm_twiddle, norm_add, norm_sub,
                                              stage, bfly_idx, q=q)


# ------------------------------------------------------
# Aval arrays (unchanged from original structure)
# ------------------------------------------------------
Aval_01 = [[] for _ in range(512)]
Aval_12 = [[] for _ in range(512)]
Aval_23 = [[] for _ in range(512)]
Aval_34 = [[] for _ in range(512)]
Aval_45 = [[] for _ in range(512)]
Aval_56 = [[] for _ in range(512)]
Aval_67 = [[] for _ in range(512)]
Aval_78 = [[] for _ in range(512)]
Aval_89 = [[] for _ in range(512)]



Aval_dict = {
    "Aval_01": Aval_01,
    "Aval_12": Aval_12,
    "Aval_23": Aval_23,
    "Aval_34": Aval_34,
    "Aval_45": Aval_45,
    "Aval_56": Aval_56,
    "Aval_67": Aval_67,
    "Aval_78": Aval_78,
    "Aval_89": Aval_89,
}

# ---------------------------------
# Helper: convert to normal domain
# ---------------------------------
def convert_to_normal_domain(gmb_mont, q=12289, Rinv=mod_inverse(2**16, 12289)):
    """
    Convert Montgomery-domain GMb twiddles to the normal domain.
    """
    return [(x * Rinv) % q for x in gmb_mont]

gmb_normal = convert_to_normal_domain(GMb_mont)

# ---------------------------------
# CSV helpers (same signatures)
# ---------------------------------
def write_forward_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, compression=CSV_COMPRESSION)

def read_forward_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression=CSV_COMPRESSION)

# ----------------------------
# Index helpers for NTT layout
# ----------------------------
def get_butterfly_input_indices(stage_idx, group_idx, butterfly_idx):
    """
    Return (input_a_index, input_b_index) for (stage, group, butterfly offset).
    """
    N = 512
    group_size = N // (2 ** stage_idx)
    stride = group_size // 2
    start = group_idx * group_size
    input_a = start + butterfly_idx
    input_b = input_a + stride
    return input_a, input_b

def get_twiddle_index(stage_idx, group_idx):
    """
    Map (stage, group) -> GMb index (Cooley–Tukey DIF layout).
    """
    return (1 << stage_idx) + group_idx

# ----------------------------------------------------------
# GPU butterfly: vectorized pair enumeration with CuPy
# ----------------------------------------------------------
import time

def butterfly_operation_strict_gpu(a1_range, a2_range, w, norm_twiddle, norm_add, norm_sub,
                                   stage, bfly_idx, q=12289):
    """
    GPU path (timed):
    - Measure 'compute' on GPU (incl. twiddle/add/sub + mask combine) with a CUDA sync
    - Measure 'df+save' on CPU (host transfer + DataFrame build + CSV writes)
    - Print timings for each butterfly
    """
    # --------- Prepare & constants ----------
    t_compute_start = time.perf_counter()

    a1 = cp.asarray(a1_range, dtype=cp.int64).reshape(-1, 1)   # (A, 1)
    a2 = cp.asarray(a2_range, dtype=cp.int64).reshape(1, -1)   # (1, B)
    A, B = a1.shape[0], a2.shape[1]
    w_dev = cp.int64(w)
    q_dev = cp.int64(q)

    # ---- Twiddle (column) ----
    raw_tw_col = a2 * w_dev                     # (1, B)
    red_tw_col = cp.mod(raw_tw_col, q_dev)      # (1, B)
    tw_norm_col = (raw_tw_col != red_tw_col)    # (1, B) booleans

    # Early mask on twiddle flag (still (1,B))
    mask_tw_col = tw_norm_col if bool(norm_twiddle) else ~tw_norm_col
    # (We still time the early exit branches, too.)
    if not mask_tw_col.any():
        # End compute timing before host work
        cp.cuda.Stream.null.synchronize()
        t_compute_end = time.perf_counter()

        # df + save timing (empty df)
        t_io_start = time.perf_counter()
        trace_df = pd.DataFrame(columns=[
            "a1","a2","Twiddled","Raw Tw","Tw Norm",
            "Raw Add","a1'","Add Norm","Raw Sub","a2'","Sub Norm"
        ])
        _save_forward_csvs(trace_df, stage, bfly_idx)
        t_io_end = time.perf_counter()

        print(f"[S{stage} B{bfly_idx}] compute: {(t_compute_end - t_compute_start)*1000:.2f} ms ")
        return trace_df

    # ---- Broadcast to (A,B) ----
    raw_tw = cp.broadcast_to(raw_tw_col, (A, B))
    red_tw = cp.broadcast_to(red_tw_col, (A, B))
    tw_norm = cp.broadcast_to(tw_norm_col, (A, B))
    mask = cp.broadcast_to(mask_tw_col, (A, B))

    # ---- Add/Sub on (A,B) ----
    raw_add = a1 + red_tw
    red_add = cp.mod(raw_add, q_dev)
    add_norm = (raw_add != red_add)

    raw_sub = a1 - red_tw
    red_sub = cp.mod(raw_sub, q_dev)
    sub_norm = (raw_sub != red_sub)

    # ---- Final mask (NO in-place) ----
    mask = mask & (add_norm == bool(norm_add)) & (sub_norm == bool(norm_sub))

    # End GPU compute timing (ensure kernels finished)
    cp.cuda.Stream.null.synchronize()
    t_compute_end = time.perf_counter()

    survivors = int(cp.asnumpy(mask).sum())  # small transfer for the count

    if survivors == 0:
        # df + save timing (empty df)
        t_io_start = time.perf_counter()
        trace_df = pd.DataFrame(columns=[
            "a1","a2","Twiddled","Raw Tw","Tw Norm",
            "Raw Add","a1'","Add Norm","Raw Sub","a2'","Sub Norm"
        ])
        _save_forward_csvs(trace_df, stage, bfly_idx)
        t_io_end = time.perf_counter()
        return trace_df

    # --------- Host transfer + DataFrame + save (timed as one block) ----------
    t_io_start = time.perf_counter()

    a1_full = cp.broadcast_to(a1, (A, B))
    a2_full = cp.broadcast_to(a2, (A, B))

    a1_vals      = cp.asnumpy(a1_full[mask])
    a2_vals      = cp.asnumpy(a2_full[mask])
    red_tw_vals  = cp.asnumpy(red_tw[mask])
    raw_tw_vals  = cp.asnumpy(raw_tw[mask])
    tw_norm_vals = cp.asnumpy(tw_norm[mask])

    raw_add_vals = cp.asnumpy(raw_add[mask])
    red_add_vals = cp.asnumpy(red_add[mask])
    add_norm_vals= cp.asnumpy(add_norm[mask])

    raw_sub_vals = cp.asnumpy(raw_sub[mask])
    red_sub_vals = cp.asnumpy(red_sub[mask])
    sub_norm_vals= cp.asnumpy(sub_norm[mask])

    trace_df = pd.DataFrame({
        "a1": a1_vals.astype(np.int64),
        "a2": a2_vals.astype(np.int64),
        "Twiddled": red_tw_vals.astype(np.int64),
        "Raw Tw": raw_tw_vals.astype(np.int64),
        "Tw Norm": tw_norm_vals.astype(np.int8),
        "Raw Add": raw_add_vals.astype(np.int64),
        "a1'": red_add_vals.astype(np.int64),
        "Add Norm": add_norm_vals.astype(np.int8),
        "Raw Sub": raw_sub_vals.astype(np.int64),
        "a2'": red_sub_vals.astype(np.int64),
        "Sub Norm": sub_norm_vals.astype(np.int8),
    })

    _save_forward_csvs(trace_df, stage, bfly_idx)

    t_io_end = time.perf_counter()

    return trace_df


# ------------------------------------------------------
# Load measured normalization flags (as before)
# ------------------------------------------------------
def load_norm_flags_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.set_index(['stage', 'butterfly_idx'], inplace=True)
    return df

norm_flag_df = load_norm_flags_from_csv("reduc_flags_GAUSS.csv")
norm_flag_df = norm_flag_df.astype({'twiddle_norm': bool, 'add_norm': bool, 'sub_norm': bool})

# ------------------------------------------------------
# Stage runner (optional multi-butterfly parallelism)
# ------------------------------------------------------
def run_forward_stage(stage_idx,
                      butterfly_count,
                      GMb_array,
                      Aval_dict,
                      q=12289,
                      stage0_input_range=None,
                      initial_input_array=None):
    """
    Run one NTT stage of guess-and-prune (GPU path called inside butterfly_operation_strict).

    Inputs
    ------
    stage_idx : int
        Stage number in [0..8].
    butterfly_count : int
        Always 256 for N=512.
    GMb_array : sequence[int]
        Twiddle factors in the *normal* domain (length >= 512).
    Aval_dict : dict[str, list[list[int]]]
        In-memory containers for value-sets between stages, e.g., "Aval_01", "Aval_12", ...
        Each entry is a list of length 512; each element is a list of candidate values.
    q : int
        Modulus (default 12289).
    stage0_input_range : list[int] | range | None
        If stage_idx == 0 and initial_input_array is None, use this list for all 512 inputs.
    initial_input_array : list[list[int]] | None
        If provided at stage 0, must be length 512; each entry is its own candidate list.

    Behavior
    --------
    - For stage 0: builds the initial candidate sets (either from `initial_input_array` or `stage0_input_range`).
    - For stage > 0: reads candidates from Aval_{stage-1}{stage}.
    - For each butterfly:
        * Locates its two input indices (ia, ib) in the stage’s wiring.
        * Reads measured flags (twiddle, add, sub) from `norm_flag_df[(stage, bfly)]`.
        * Calls `butterfly_operation_strict(...)` (which performs GPU vectorization).
        * Updates the next-stage candidate sets for indices ia and ib from the surviving outputs a1' and a2'.
        * CSVs are written inside `butterfly_operation_strict`.

    Notes
    -----
    - This function assumes CuPy/CUDA is required and already validated at import time.
    - No cross-butterfly threading here (simplest and easiest to follow). The heavy work
      is vectorized on the GPU *inside* each butterfly call.
    """

    # Determine keys for previous and next Aval (e.g., stage 2 -> prev "Aval_12", next "Aval_23")
    PREV = stage_idx - 1
    CURR = stage_idx
    POST = stage_idx + 1

    prev_key = f"Aval_{PREV}{CURR}"
    next_key = f"Aval_{CURR}{POST}" if stage_idx < 8 else None  # No next array at final stage

    # -------------------------------
    # Provision input candidate sets
    # -------------------------------
    if stage_idx == 0:
        # Option A: explicit per-index initial_input_array (length 512)
        if initial_input_array is not None:
            if len(initial_input_array) != 512:
                raise ValueError("initial_input_array must have length 512 for stage 0.")
            input_array = initial_input_array

        # Option B: same range for all 512 indices
        elif stage0_input_range is not None:
            base = list(stage0_input_range)
            input_array = [base[:] for _ in range(512)]  # copy to avoid shared list aliasing

        else:
            raise ValueError("For stage 0, provide either stage0_input_range or initial_input_array.")
    else:
        # Read the previous stage’s outputs as our inputs
        if prev_key not in Aval_dict:
            raise KeyError(f"Missing '{prev_key}' in Aval_dict for stage {stage_idx}.")
        input_array = Aval_dict[prev_key]

    # Next-stage output array (none at final stage)
    output_array = Aval_dict[next_key] if next_key else None

    # ------------------------------
    # Stage wiring / indexing math
    # ------------------------------
    # There are 2^stage groups; each group covers (512 / 2^stage) indices.
    group_count = 2 ** stage_idx
    group_size = 512 // group_count
    butterflies_per_group = group_size // 2  # each butterfly consumes a pair (stride = group_size/2)

    # ------------------------------
    # Main butterfly loop
    # ------------------------------
    for bfly_idx in range(butterfly_count):
        # Identify which group this butterfly belongs to and its offset inside that group
        group_idx   = bfly_idx // butterflies_per_group
        bfly_offset = bfly_idx %  butterflies_per_group

        # Resolve the two input indices (ia, ib) wired into this butterfly at this stage
        ia, ib = get_butterfly_input_indices(stage_idx, group_idx, bfly_offset)

        # Twiddle factor for this (stage, group) in normal domain
        tf_index = get_twiddle_index(stage_idx, group_idx)
        w = GMb_array[tf_index]

        # Measured normalization flags (bools) for this (stage, butterfly)
        try:
            norm_tw, norm_add, norm_sub = norm_flag_df.loc[
                (stage_idx, bfly_idx), ['twiddle_norm', 'add_norm', 'sub_norm']
            ]
        except KeyError:
            raise KeyError(f"Missing normalization flags for (stage={stage_idx}, butterfly={bfly_idx}).")

        # Candidate inputs for this butterfly’s two wires
        a1_range = input_array[ia]
        a2_range = input_array[ib]

        # Sanity: empty inputs imply prior pruning removed all options
        if not a1_range or not a2_range:
            # Still produce empty CSVs via the call for consistency, but it will short-circuit quickly.
            print(f"[WARNING] Empty inputs at stage {stage_idx}, butterfly {bfly_idx} (ia={ia}, ib={ib}).")

        # ---- GPU-heavy work happens inside this call ----
        df = butterfly_operation_strict(
            a1_range=a1_range,
            a2_range=a2_range,
            w=w,
            norm_twiddle=bool(norm_tw),
            norm_add=bool(norm_add),
            norm_sub=bool(norm_sub),
            stage=stage_idx,
            bfly_idx=bfly_idx,
            q=q
        )

        if df.empty:
            print(f"[WARNING] Pruned all guesses at stage {stage_idx}, butterfly {bfly_idx}")

        # Update next-stage candidates (unless we are at the final stage)
        if output_array is not None:
            # Unique values that satisfied constraints for each output wire
            a1_prime_vals = df["a1'"].unique().tolist() if not df.empty else []
            a2_prime_vals = df["a2'"].unique().tolist() if not df.empty else []

            output_array[ia] = a1_prime_vals
            output_array[ib] = a2_prime_vals

        # Progress trace
        print(f"[S{stage_idx}] Butterfly {bfly_idx} done.")


# --------------------------------------------
# Run (same as original sequence; tweak)
# --------------------------------------------
if __name__ == "__main__":
    gmb_normal = convert_to_normal_domain(GMb_mont)

    # Stage 0: mixed range [0..10] ∪ [12280..12288]
    init_range = list(range(0, 11)) + list(range(12280, 12289))

    run_forward_stage(stage_idx=0, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289, stage0_input_range=init_range)
    run_forward_stage(stage_idx=1, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289)
    run_forward_stage(stage_idx=2, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289)
    run_forward_stage(stage_idx=3, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289)
    run_forward_stage(stage_idx=4, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289)
    run_forward_stage(stage_idx=5, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289)
    run_forward_stage(stage_idx=6, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289)
    run_forward_stage(stage_idx=7, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289)
    run_forward_stage(stage_idx=8, butterfly_count=256, GMb_array=gmb_normal, Aval_dict=Aval_dict, q=12289)

