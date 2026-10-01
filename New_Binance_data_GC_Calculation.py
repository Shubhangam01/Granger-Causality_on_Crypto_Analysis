import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.multitest import multipletests  # For BH Correction
from multiprocessing import Pool, cpu_count
import contextlib
import io

# === Parameters ===
input_dir = r"D:\binance_data\dane_Binance140\logret_windows"
output_dir = r"D:\New_GC_Matrices_Binance_data"
os.makedirs(output_dir, exist_ok=True)

# === Optimal Lag Selection using BIC ===
def optimal_lag_bic(X, max_lag=10):
    try:
        model = VAR(X)
        result = model.select_order(maxlags=max_lag)
        return result.bic.idxmin() if hasattr(result.bic, 'idxmin') else result.bic
    except:
        return -1


def compute_gc_strength(x, y, lag):
    try:
        model_full = VAR(np.column_stack([y, x])).fit(lag)
        sigma2_full = model_full.sigma_u[0, 0]
        model_restricted = AutoReg(y, lags=lag, old_names=False).fit()
        sigma2_restricted = np.var(model_restricted.resid)
        return np.log(sigma2_restricted / sigma2_full)
    except:
        return 0.0

# === Worker for Pairwise GC Calculation ===
def process_pair(args):
    i, j, data, cryptos, max_lag, alpha = args
    if i == j:
        return (i, j, 1.0, -1)  # Return p-value of 1.0 (fail) and dummy lag
    
    x = data[i]
    y = data[j]
    X = np.column_stack([y, x])
    lag = optimal_lag_bic(X, max_lag=max_lag)
    
    # Adjust lag constraint dynamically based on available data points
    if lag < 1 or lag > min(10, len(x) // 5):
        return (i, j, 1.0, -1)
    
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            test_result = grangercausalitytests(X, maxlag=lag)
        p_value = test_result[lag][0]['ssr_ftest'][1]
        
        # Return the raw p-value and lag to be corrected later
        return (i, j, p_value, lag)
    except:
        pass
    return (i, j, 1.0, -1)

# === Compute GC Matrix WITHOUT multiprocessing ===
def granger_matrix_for_week(logret_df, max_lag=10, alpha=0.05):
    cryptos = logret_df.index.tolist()
    data = logret_df.values
    n = len(cryptos)
    gc_matrix = pd.DataFrame(0.0, index=cryptos, columns=cryptos)
    args_list = [(i, j, data, cryptos, max_lag, alpha) for i in range(n) for j in range(n) if i != j]

    # Serial execution — no nested multiprocessing
    results = [process_pair(args) for args in args_list]

    # === Benjamini-Hochberg FDR Correction ===
    # Extract all p-values from the processed pairs
    p_values = [res[2] for res in results]
    
    if len(p_values) > 0:
        # reject_mask is a boolean array of pairs that survived the FDR correction
        reject_mask, _, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    else:
        reject_mask = [False] * len(p_values)

    # Assign GC strength only to the links that survived the BH correction
    for k, (i, j, p_val, lag) in enumerate(results):
        if reject_mask[k] and lag != -1:
            gc_strength = compute_gc_strength(data[i], data[j], lag)
            gc_matrix.iloc[i, j] = gc_strength

    return gc_matrix

# === File-level Worker ===
def process_single_file(file_args):
    file, input_dir, output_dir, max_lag, alpha = file_args
    input_path = os.path.join(input_dir, file)
    output_name = file.replace("logretwin_", "GC_matrix_")
    output_path = os.path.join(output_dir, output_name)
    try:
        # Read the file setting the timestamp column as the index
        logret_df = pd.read_csv(input_path, index_col=0)
        
        # Transpose the dataframe so cryptos are rows and timestamps are columns
        logret_df = logret_df.T
        
        # Sort cryptos alphabetically to maintain deterministic matrix ordering
        logret_df = logret_df.sort_index(axis=0)
        
        gc_matrix = granger_matrix_for_week(logret_df, max_lag=max_lag, alpha=alpha)
        gc_matrix.to_csv(output_path, float_format='%.15f')
        return True
    except Exception as e:
        print(f"❌ Failed processing {file}: {e}")
        return False

# === Batch Process Remaining Files in Parallel ===
def process_remaining_weeks(input_dir, output_dir, max_lag=10, alpha=0.05):
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    
    # Check output directory using the updated filename prefixes
    processed_files = set([f.replace("GC_matrix_", "logretwin_") for f in os.listdir(output_dir) if f.endswith('.csv')])
    remaining_files = [f for f in input_files if f not in processed_files]

    print(f"🔍 {len(remaining_files)} files remaining out of {len(input_files)} total.\n")

    file_args_list = [(f, input_dir, output_dir, max_lag, alpha) for f in remaining_files]

    with Pool(processes=min(cpu_count(), len(remaining_files))) as pool:
        list(tqdm(pool.imap_unordered(process_single_file, file_args_list), total=len(remaining_files), desc="⚡ Parallel File Processing"))

    print("\n✅ Remaining GC matrices processed and saved.")

# === Main Runner ===
if __name__ == "__main__":
    process_remaining_weeks(input_dir, output_dir, max_lag=10, alpha=0.05)