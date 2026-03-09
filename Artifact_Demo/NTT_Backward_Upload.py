import pandas as pd

# Use zip for broad compatibility and strong size reduction.
CSV_COMPRESSION = "zip" 

def get_csv_indices_and_column(stage, bfly_idx):
    """
    Compute the two .csv indices and the column ('a1' or 'a2') to use
    for backward inference at a given stage and butterfly index.

    Args:
        stage (int): stage index (1 to 8), where stage s means inference from s+1 → s
        bfly_idx (int): butterfly index (0 to 255)

    Returns:
        (list[int], str): [left_index, right_index], and column name 'a1' or 'a2'
    """
    assert 0 <= stage <= 9, "Stage must be in [0, 8]"
    assert 0 <= bfly_idx < 256, "Butterfly index must be in [0, 255]"

    group_size = 2 ** (9-stage)
    group_start = (bfly_idx // group_size) * group_size
    group_middle = group_start + (group_size // 2)

    if bfly_idx < group_middle:
        left_idx = bfly_idx
        right_idx = bfly_idx + (group_size // 2)
        column = 'a1'
    else:
        right_idx = bfly_idx
        left_idx = bfly_idx - (group_size // 2)
        column = 'a2'

    return [left_idx, right_idx], column

def write_forward_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a zip-compressed CSV, keeping the same filename."""
    df.to_csv(path, index=False, compression=CSV_COMPRESSION)

def read_forward_csv(path: str) -> pd.DataFrame:
    """Read a zip-compressed CSV that uses the original filename."""
    return pd.read_csv(path, compression=CSV_COMPRESSION)

def backward_trace_single_butterfly(refined_csv_a1, refined_csv_a2, butterfly_trace_csv, column, stage_idx):
    """
    Backward filters a butterfly operation based on refined outputs a1' and a2'.

    Parameters:
    - refined_csv_a1: CSV file containing refined values (use column = 'a1' or 'a2')
    - refined_csv_a2: CSV file containing refined values (use the same column)
    - butterfly_trace_csv: full trace CSV containing columns [a1, a2, a1', a2'] from forward path
    - column: which column ('a1' or 'a2') to read from the refined CSVs

    Returns:
    - refined_a1, refined_a2: sorted list of valid a1 and a2 after filtering
    """
    # Step 1: Load refined output values (they appear as 'a1' or 'a2' in those CSVs)
    df_a1p = read_forward_csv(refined_csv_a1)
    df_a2p = read_forward_csv(refined_csv_a2)

    refined_a1p = set(df_a1p[column].dropna().astype(int).unique())
    refined_a2p = set(df_a2p[column].dropna().astype(int).unique())

    # Step 2: Load butterfly trace
    df_full = read_forward_csv(butterfly_trace_csv)
    original_a1_count = df_full["a1"].nunique()
    original_a2_count = df_full["a2"].nunique()

    # Step 3: Filter rows by refined outputs
    df_filtered = df_full[
        df_full["a1'"].astype(int).isin(refined_a1p) &
        df_full["a2'"].astype(int).isin(refined_a2p)
    ]

    filtered_a1_count = df_filtered["a1"].nunique() 
    filtered_a2_count = df_filtered["a2"].nunique() 
    print("filtering completed")

    # Step 4: Extract refined a1 and a2
    refined_a1 = sorted(df_filtered["a1"].dropna().astype(int).unique())
    refined_a2 = sorted(df_filtered["a2"].dropna().astype(int).unique())

    # Step 5: Overwrite the original butterfly trace file
    write_forward_csv(df_filtered, butterfly_trace_csv)

    return filtered_a1_count, filtered_a2_count


def run_backward_stage(stage_idx, butterfly_count, summarize_stage0=False):
    """
    Runs backward tracing for all butterflies in a given NTT stage.

    Parameters:
    - stage_idx: integer in [1..8]; stage s means we refine stage s using constraints from stage s+1
    - butterfly_count: number of butterflies in this stage (256 for N=512)
    """
    # stage_idx += 1
    assert 0 <= stage_idx <= 8, "run_backward_stage expects stage_idx in [0, 8]"
    # assert 0 <= i <= 9, "run_backward_stage expects stage_idx in [0, 8]"

    total_after_counts = 0 if (summarize_stage0 and stage_idx == 0) else None

    later_stage = stage_idx + 1  # consume refined outputs from the next stage
    for bfly_idx in range(butterfly_count):
        key = f"stage{stage_idx}_butterfly{bfly_idx}"
        stage_idx = stage_idx

        # Derive the two CSV indices and which column to read from them
        (left_idx, right_idx), column = get_csv_indices_and_column(later_stage, bfly_idx)
        
        # print(left_idx)
        # print(right_idx)

        # Files from the later stage representing a1' and a2' (stored under 'column')
        refined_a1p_csv = f"C:/Users/jqiu2/Desktop/Artifact_Demo/Data/GPU/for_Backward/for_backward_stage{later_stage}_butterfly{left_idx}.csv"
        refined_a2p_csv = f"C:/Users/jqiu2/Desktop/Artifact_Demo/Data/GPU/for_Backward/for_backward_stage{later_stage}_butterfly{right_idx}.csv"

        # This butterfly's original trace file (from forward step)
        butterfly_trace_csv = f"C:/Users/jqiu2/Desktop/Artifact_Demo/Data/GPU/for_Backward/for_backward_stage{stage_idx}_butterfly{bfly_idx}.csv"

        print(f"\n=== Backward Tracing {key} ===")
        c_a1, c_a2 = backward_trace_single_butterfly(
            refined_csv_a1=refined_a1p_csv,
            refined_csv_a2=refined_a2p_csv,
            butterfly_trace_csv=butterfly_trace_csv,
            column=column,
            stage_idx = stage_idx
        )

        if total_after_counts is not None:
            total_after_counts += (c_a1 + c_a2)
    if total_after_counts is not None:
        denom = 21 * 512
        ratio = total_after_counts / denom
        print("\n[Summary @ Stage 0]")
        print(f"[Total After] Sum of 'after' counts over all 512 variables: {total_after_counts}")
        print(f"[Normalized] Total / (21×512 = {denom}) = {ratio:.6f}")
        return total_after_counts, ratio


# run_backward_stage(stage_idx=8, butterfly_count=256, summarize_stage0=True)
# run_backward_stage(stage_idx=7, butterfly_count=256, summarize_stage0=True)
# run_backward_stage(stage_idx=6, butterfly_count=256, summarize_stage0=True)
# run_backward_stage(stage_idx=5, butterfly_count=256, summarize_stage0=True)
# run_backward_stage(stage_idx=4, butterfly_count=256, summarize_stage0=True)
# run_backward_stage(stage_idx=3, butterfly_count=256, summarize_stage0=True)
# run_backward_stage(stage_idx=2, butterfly_count=256, summarize_stage0=True)
# run_backward_stage(stage_idx=1, butterfly_count=256, summarize_stage0=True)
run_backward_stage(stage_idx=0, butterfly_count=256, summarize_stage0=True)



