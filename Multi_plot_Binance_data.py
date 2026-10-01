import os
import re
import glob
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# 1. USER CONFIGURATIONS (Full Control over Aesthetics & Parameters)
# =============================================================================
# Directories
GC_DIR = r"D:\New_GC_Matrices_Binance_data"
PRICE_DIR = r"D:\binance_data\dane_Binance140\price_windows"
OUTPUT_DIR = r"D:\Sem_Project"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text and Labels
TITLE_SIG = ""
TITLE_NONSIG = ""
Y_LABEL_CORR = "Correlation Magnitude (|r|)"
Y_LABEL_PVAL = "Significance"

# Fonts
FONT_FAMILY = "serif"        # Options: 'serif', 'sans-serif', 'monospace'
TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 36
TICK_FONT_SIZE_Y = 26
TICK_FONT_SIZE_X = 18      
LEGEND_FONT_SIZE = 20

# Bar Aesthetics
BAR_THICKNESS = 0.4          # Width of the bars
COLOR_COINS = "#FFC300"      # Yellow/Gold
COLOR_TOKENS = "#228B22"     # Forest Green
BAR_EDGE_COLOR = "black"     
BAR_EDGE_WIDTH = 0.3

# Output Aesthetics
IMAGE_DPI = 600              # Publication quality resolution
FIG_HEIGHT = 16              # Height for dual panels

# =============================================================================
# 2. TARGET CRYPTO LISTS
# =============================================================================
COIN_LIST = [
    "ADA", "ALGO", "ARDR", "ATOM", "AVAX", "BAND", "BCH", "BNB", "BTC", "CELO",
    "CHR", "CHZ", "COS", "COTI", "CTK", "CTXC", "DASH", "DCR", "DGB", "DOGE",
    "DOT", "DUSK", "EGLD", "EOS", "ETC", "ETH", "FET", "FIL", "FIO", "FTM",
    "HBAR", "HIVE", "ICX", "INJ", "IOST", "IOTA", "IOTX", "IRIS", "KAVA", "KMD",
    "KSM", "LSK", "LTC", "LTO", "NEAR", "NEO", "NKN", "NULS", "ONE", "ONG",
    "ONT", "QTUM", "ROSE", "RUNE", "RVN", "SC", "SOL", "STX", "SXP", "TFUEL",
    "THETA", "TRX", "VET", "VITE", "VTHO", "WAN", "XLM", "XRP", "XTZ", "ZEC",
    "ZEN", "ZIL"
]

TOKEN_LIST = [
    "1INCH", "AAVE", "AKRO", "ALPHA", "ANKR", "ARPA", "ASR", "ATM", "AUDIO", 
    "AVA", "AXS", "BAL", "BAT", "BEL", "BLZ", "BNT", "CELR", "COMP", "CRV", 
    "CTSI", "DATA", "DENT", "DIA", "ENJ", "FLM", "FUN", "GRT", "HARD", "JST", 
    "JUV", "KNC", "LINK", "LRC", "MANA", "MBL", "MDT", "MKR", "MTL", "NMR", 
    "OG", "OGN", "ORN", "OXT", "PSG", "REN", "RLC", "RSR", "SAND", "SKL", 
    "SNX", "STMX", "STORJ", "STPT", "SUSHI", "TRB", "TROY", "UMA", "UNFI", 
    "UNI", "UTK", "WIN", "WING", "WRX", "XVS", "YFI", "ZRX", "PAXG"
]

# =============================================================================
# 3. DATA PROCESSING & CORRELATION ENGINE
# =============================================================================
def calculate_correlations():
    print("=" * 90)
    print("EXTRACTING DATA AND CALCULATING CORRELATIONS...")
    print("=" * 90)
    
    def extract_date(filepath):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(filepath))
        return match.group(1) if match else None

    gc_files = glob.glob(os.path.join(GC_DIR, "*.csv"))
    price_files = glob.glob(os.path.join(PRICE_DIR, "*.csv"))

    gc_map = {extract_date(f): f for f in gc_files if extract_date(f)}
    price_map = {extract_date(f): f for f in price_files if extract_date(f)}
    common_dates = sorted(list(set(gc_map.keys()) & set(price_map.keys())))
    
    if not common_dates:
        print("❌ No matching weeks found between GC matrices and Price Windows!")
        return None, None, None, None

    all_cryptos = COIN_LIST + TOKEN_LIST
    crypto_data = {crypto: {'outstrength': [], 'returns': []} for crypto in all_cryptos}

    for date_str in common_dates:
        try:
            gc_matrix = pd.read_csv(gc_map[date_str], index_col=0)
            gc_matrix.index = gc_matrix.index.astype(str).str.strip().str.upper()
            outstrengths = gc_matrix.sum(axis=1)

            price_df = pd.read_csv(price_map[date_str], index_col=0)
            price_df.columns = price_df.columns.astype(str).str.strip().str.upper()

            for crypto in all_cryptos:
                if crypto in outstrengths.index and crypto in price_df.columns:
                    valid_prices = price_df[crypto].dropna()
                    if len(valid_prices) > 0:
                        first_price, last_price = valid_prices.iloc[0], valid_prices.iloc[-1]
                        if first_price > 0 and last_price > 0:
                            crypto_data[crypto]['outstrength'].append(outstrengths[crypto])
                            crypto_data[crypto]['returns'].append(np.log(last_price / first_price))
        except Exception:
            continue

    # Categorize results
    sig_coins, nonsig_coins = [], []
    sig_tokens, nonsig_tokens = [], []

    for crypto in all_cryptos:
        data = crypto_data[crypto]
        valid_pairs = [(x, y) for x, y in zip(data['outstrength'], data['returns']) if pd.notna(x) and pd.notna(y)]
        
        if len(valid_pairs) >= 10:
            x_vals = [p[0] for p in valid_pairs]
            y_vals = [p[1] for p in valid_pairs]
            r, p = stats.pearsonr(x_vals, y_vals)
            
            result_tuple = (crypto, r, p)
            is_coin = crypto in COIN_LIST
            
            if p < 0.01:
                (sig_coins if is_coin else sig_tokens).append(result_tuple)
            else:
                (nonsig_coins if is_coin else nonsig_tokens).append(result_tuple)

    print(f"✅ Calculation Complete: {len(sig_coins)} Sig Coins | {len(sig_tokens)} Sig Tokens")
    return sig_coins, sig_tokens, nonsig_coins, nonsig_tokens


# =============================================================================
# 4. PLOTTING FUNCTIONS
# =============================================================================
def prepare_grouped_data(data_lists, type_labels, color_map, is_sig=True):
    df_list = []
    for data, label in zip(data_lists, type_labels):
        temp_df = pd.DataFrame(data, columns=["crypto", "r", "p"])
        temp_df["type"] = label
        
        # Clean ticker just in case
        temp_df["crypto"] = temp_df["crypto"].astype(str).str.replace('USD', '', regex=False)
        temp_df["neg_log_p"] = -np.log10(temp_df["p"])
        df_list.append(temp_df)
        
    full_df = pd.concat(df_list).reset_index(drop=True)
    final_dfs = []
    group_info = [] 
    
    if is_sig:
        for label in type_labels:
            group = full_df[full_df['type'] == label].copy()
            if not group.empty:
                group = group.sort_values(by="r", ascending=True)
                group['plot_val'] = group['r'].abs()
                final_dfs.append(group)
                group_info.append((label, len(group), color_map[label]))
    else:
        for label in type_labels:
            group = full_df[(full_df['type'] == label) & (full_df['r'] < 0)].copy()
            if not group.empty:
                group = group.sort_values(by="r", ascending=True)
                group['plot_val'] = group['r']
                final_dfs.append(group)
                group_info.append((f"{label}\n(r < 0)", len(group), color_map[label]))
        for label in type_labels:
            group = full_df[(full_df['type'] == label) & (full_df['r'] >= 0)].copy()
            if not group.empty:
                group = group.sort_values(by="r", ascending=False)
                group['plot_val'] = group['r']
                final_dfs.append(group)
                group_info.append((f"{label}\n(r > 0)", len(group), color_map[label]))
                
    if not final_dfs:
        return pd.DataFrame(), []
        
    df_sorted = pd.concat(final_dfs).reset_index(drop=True)
    df_sorted['color'] = df_sorted['type'].map(color_map)
    return df_sorted, group_info

def build_dual_panel_plot(df, group_info, is_sig_plot=True):
    if df.empty:
        print(f"⚠️ No data to plot for {'Significant' if is_sig_plot else 'Non-Significant'} assets.")
        return

    dynamic_width = max(30, len(df) * 0.25)
    plt.rcParams["font.family"] = FONT_FAMILY
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(dynamic_width, FIG_HEIGHT), 
                                   gridspec_kw={'height_ratios': [2, 1]}, sharex=False)

    for i in range(len(df)):
        if i % 2 == 0:
            ax1.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.08, zorder=0)
            ax2.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.08, zorder=0)

    ax1.bar(df['crypto'], df['plot_val'], width=BAR_THICKNESS, color=df['color'], 
            edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    
    ax1.set_title(TITLE_SIG if is_sig_plot else TITLE_NONSIG, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=30)
    y_label = Y_LABEL_CORR if is_sig_plot else "Pearson Correlation (r)"
    ax1.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=15)
    ax1.grid(axis='y', linestyle=':', alpha=0.6, zorder=1)
    
    if is_sig_plot:
        y_max = df['plot_val'].max()
        ax1.set_ylim(0, y_max * 1.25)
        text_y_pos = y_max * 1.1
    else:
        y_min, y_max = df['plot_val'].min(), df['plot_val'].max()
        y_max = 0.1 if y_max <= 0 else y_max
        span = y_max - y_min
        ax1.set_ylim(y_min - 0.1 * span, y_max + 0.3 * span)
        text_y_pos = y_max + 0.15 * span
        ax1.axhline(0, color='black', linewidth=1.5, zorder=2)

    ax1.set_xlim(-1, len(df))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    cumulative_idx = 0
    for i, (label, count, color) in enumerate(group_info):
        center_x = cumulative_idx + (count / 2) - 0.5
        ax1.text(center_x, text_y_pos, f"{label.upper()}\n(n={count})", 
                 ha='center', va='bottom', fontsize=LABEL_FONT_SIZE, fontweight='bold', color=color)
        cumulative_idx += count
        if i < len(group_info) - 1:
            ax1.axvline(x=cumulative_idx - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)
            ax2.axvline(x=cumulative_idx - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)

    unique_types = df['type'].unique()
    patches = [mpatches.Patch(color=COLOR_MAP[t], label=f'{t}') for t in unique_types]
    legend_title = "Asset Class (Magnitudes Plotted)" if is_sig_plot else "Asset Class (Actual r Plotted)"
    legend = ax1.legend(handles=patches, fontsize=LEGEND_FONT_SIZE, loc='upper right', framealpha=0.95,
                        title=legend_title, title_fontsize=LEGEND_FONT_SIZE)
    legend.get_title().set_fontweight('bold')

    ax2.bar(df['crypto'], df['neg_log_p'], width=BAR_THICKNESS, color=df['color'], 
            edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    ax2.set_ylabel(Y_LABEL_PVAL, fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=15)
    ax2.grid(axis='y', linestyle=':', alpha=0.6, zorder=1)
    
    ax2.axhline(2.0, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=4, label="p = 0.01 Threshold")
    ax2.text(-0.5, 2.1, "p = 0.01", color='red', fontsize=12, fontweight='bold', zorder=5)
    ax2.set_xlim(-1, len(df))
    
    if is_sig_plot:
        ax2.set_ylim(2, math.ceil(df['neg_log_p'].max()) + 1)
    else:
        ax2.set_ylim(0, 2.5)
        
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for ax in [ax1, ax2]:
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['crypto'], rotation=90, ha='center', fontsize=TICK_FONT_SIZE_X)
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE_Y)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45) 

    suffix = "Significant" if is_sig_plot else "NonSignificant"
    pdf_path = os.path.join(OUTPUT_DIR, f"Correlation_DualPanel_{suffix}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"Correlation_DualPanel_{suffix}.png")
    
    plt.savefig(pdf_path, format='pdf', dpi=IMAGE_DPI, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=IMAGE_DPI, bbox_inches='tight')
    print(f"✅ [{suffix}] plots saved to:\n -> {pdf_path}\n -> {png_path}\n")


# =============================================================================
# 5. EXECUTION
# =============================================================================
if __name__ == "__main__":
    COLOR_MAP = {"Coins": COLOR_COINS, "Tokens": COLOR_TOKENS}
    
    sig_coins, sig_tokens, nonsig_coins, nonsig_tokens = calculate_correlations()
    
    if sig_coins is not None:
        print("\n" + "="*90)
        print("GENERATING DUAL-PANEL PUBLICATION PLOTS...")
        print("="*90)

        # 1. Plot Significant
        df_sig, group_info_sig = prepare_grouped_data(
            data_lists=[sig_coins, sig_tokens],
            type_labels=["Coins", "Tokens"],
            color_map=COLOR_MAP,
            is_sig=True
        )
        build_dual_panel_plot(df_sig, group_info_sig, is_sig_plot=True)

        # 2. Plot Non-Significant
        df_nonsig, group_info_nonsig = prepare_grouped_data(
            data_lists=[nonsig_coins, nonsig_tokens],
            type_labels=["Coins", "Tokens"],
            color_map=COLOR_MAP,
            is_sig=False
        )
        build_dual_panel_plot(df_nonsig, group_info_nonsig, is_sig_plot=False)
        
        plt.show()