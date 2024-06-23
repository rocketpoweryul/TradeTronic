import csv
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import multiprocessing
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import functions from TATools
from TATools import (fetch_OHLCV, find_swing_high_and_lows, filter_swing_high_and_lows,
                     filter_peaks, detect_consolidation, add_moving_average,
                     add_relative_strength_line, get_stage2_uptrend,
                     calculate_up_down_volume_ratio, calculate_atr, calculate_pct_b)

from GUI import *

# Configuration and hyperparameters
SEQUENCE_LENGTH = 63
INPUT_FILE = 'data/us_equities_data.csv'
OUTPUT_SEQ_FILE = 'sequences.npy'
OUTPUT_LABELS_FILE = 'labels.npy'
OUTPUT_METADATA_FILE = 'metadata.npy'
OUTPUT_CSV_FILE = 'log/TrainingLog.csv'
MAX_WORKERS = min(11, os.cpu_count() - 1)

def calculate_consolidation_stats(df):
    """Calculate consolidation length and depth."""
    consol_lengths = []
    consol_depths = []
    current_consol_start = None
    
    for i, row in df.iterrows():
        if row['Consol_Detected'] and current_consol_start is None:
            current_consol_start = i
        elif not row['Consol_Detected'] and current_consol_start is not None:
            length = (i - current_consol_start).days
            depth = (df.loc[current_consol_start:i, 'High'].max() - df.loc[current_consol_start:i, 'Low'].min()) / df.loc[current_consol_start, 'Close']
            consol_lengths.append(length)
            consol_depths.append(depth)
            current_consol_start = None
    
    return consol_lengths, consol_depths

def count_rsl_nh(sequence):
    """Count RSL_NH occurrences in a sequence."""
    return sum(1 for row in sequence if row[-1] == 1)  # Assuming RSL_NH is the last column

def log_statistics(data, name):
    """Log statistics for a given dataset."""
    print(f"\n{name} Statistics:")
    print(f"Count: {len(data)}")
    print(f"Mean: {np.mean(data)}")
    print(f"Median: {np.median(data)}")
    print(f"Standard Deviation: {np.std(data)}")
    print(f"Variance: {np.var(data)}")
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")

def plot_distribution(data, name, filename):
    """Plot and save distribution of data."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(f"Distribution of {name}")
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()

def process_symbol(symbol):
    """Process a single stock symbol to generate sequences, profits, and metadata."""
    try:
        start_time = time.time()

        df = fetch_OHLCV(symbol, num_bars=None, interval='D')
        if df is None or df.empty:
            return [], [], [], [], [], []

        # Apply technical analysis
        df = find_swing_high_and_lows(df)
        df = filter_swing_high_and_lows(df)
        df = filter_peaks(df)
        df = detect_consolidation(df)
        df = add_moving_average(df, 21, 'ema')
        df = add_moving_average(df, 50, 'sma')
        df = add_moving_average(df, 150, 'sma')
        df = add_moving_average(df, 200, 'sma')
        df = add_relative_strength_line(df)
        df = get_stage2_uptrend(df)
        df = calculate_up_down_volume_ratio(df)
        df = calculate_atr(df)
        df = calculate_pct_b(df)

        symbol_sequences = []
        symbol_profits = []
        symbol_metadata = []
        symbol_consol_lengths, symbol_consol_depths = calculate_consolidation_stats(df)
        symbol_rsl_nh_counts = []

        for index in range(len(df) - 1):
            if not df['Consol_Detected'].iloc[index] or df['Consol_Detected'].iloc[index + 1]:
                continue

            consol_lhs_price = df['Consol_LHS_Price'].iloc[index]
            start_index = max(0, index - SEQUENCE_LENGTH + 1)
            
            seq = df.iloc[start_index:index + 1].values.tolist()
            if len(seq) < SEQUENCE_LENGTH:
                padding = [[0] * len(df.columns)] * (SEQUENCE_LENGTH - len(seq))
                seq = padding + seq
            
            for i in range(index + 1, len(df) - 2):
                if not (df['Close'].iloc[i] < df['Close_50_bar_sma'].iloc[i] and 
                        df['Close'].iloc[i + 1] < df['Close_50_bar_sma'].iloc[i + 1] and 
                        df['Close'].iloc[i + 2] < df['Close_50_bar_sma'].iloc[i + 2]):
                    continue

                profit = df['Close'].iloc[i + 2] / consol_lhs_price - 1
                
                symbol_sequences.append(seq)
                symbol_profits.append(profit)
                symbol_metadata.append([symbol, df.index[index + 1].strftime('%Y-%m-%d')])  # Use index for date
                symbol_rsl_nh_counts.append(count_rsl_nh(seq))
                break

        end_time = time.time()
        return symbol_sequences, symbol_profits, symbol_metadata, symbol_consol_lengths, symbol_consol_depths, symbol_rsl_nh_counts

    except Exception as e:
        print(f"Error processing symbol {symbol}: {e}")
        return [], [], [], [], [], []

def load_sec_list(filename):
    """Load the list of securities from a CSV file."""
    try:
        df = pd.read_csv(filename)
        sec_list = df.iloc[:, 0].tolist()
        if not sec_list:
            raise ValueError("No securities found in the input file")
        return sec_list
    except Exception as e:
        print(f"Error loading securities list: {e}")
        return []

def save_data(sequences, profits, metadata):
    """Save processed data to files."""
    np.save(OUTPUT_SEQ_FILE, sequences)
    np.save(OUTPUT_LABELS_FILE, profits)
    np.save(OUTPUT_METADATA_FILE, metadata)

    combined_data = [[meta[0], meta[1], profit] for meta, profit in zip(metadata, profits)]
    pd.DataFrame(combined_data, columns=['Symbol', 'Date', 'Profit']).to_csv(OUTPUT_CSV_FILE, index=False)

def main():
    print("Accessing list of securities...")
    try:
        print(f"Loading {INPUT_FILE} file")
        sec_list = load_sec_list(INPUT_FILE)
        print(f"Total securities: {len(sec_list)}")

        all_sequences = []
        all_profits = []
        all_metadata = []
        all_consol_lengths = []
        all_consol_depths = []
        all_rsl_nh_counts = []

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_symbol, symbol): symbol for symbol in sec_list}
            
            for future in tqdm(as_completed(futures), total=len(sec_list), desc="Processing symbols"):
                symbol = futures[future]
                try:
                    symbol_sequences, symbol_profits, symbol_metadata, symbol_consol_lengths, symbol_consol_depths, symbol_rsl_nh_counts = future.result()
                    all_sequences.extend(symbol_sequences)
                    all_profits.extend(symbol_profits)
                    all_metadata.extend(symbol_metadata)
                    all_consol_lengths.extend(symbol_consol_lengths)
                    all_consol_depths.extend(symbol_consol_depths)
                    all_rsl_nh_counts.extend(symbol_rsl_nh_counts)
                except Exception as exc:
                    print(f'{symbol} generated an exception: {exc}')

        if not all_sequences or not all_profits or not all_metadata:
            raise ValueError("No valid data processed")

        sequences_array = np.array(all_sequences)
        profits_array = np.array(all_profits)

        save_data(sequences_array, profits_array, all_metadata)

        # Log statistics
        log_statistics(all_profits, "Profit Distribution")
        log_statistics(all_consol_lengths, "Consolidation Length")
        log_statistics(all_consol_depths, "Consolidation Depth")
        log_statistics(all_rsl_nh_counts, "RSL_NH Occurrences")

        # Plot distributions
        plot_distribution(all_profits, "Profit Distribution", "profit_distribution.png")
        plot_distribution(all_consol_lengths, "Consolidation Length", "consolidation_length_distribution.png")
        plot_distribution(all_consol_depths, "Consolidation Depth", "consolidation_depth_distribution.png")
        plot_distribution(all_rsl_nh_counts, "RSL_NH Occurrences", "rsl_nh_occurrences_distribution.png")

        print("Data processing completed successfully.")

    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == '__main__':
    main()