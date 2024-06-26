import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

# Import functions from your libraries
from NorgateInterface import fetch_OHLCV
from TATools import *

# Configuration
SEQUENCE_LENGTH = 63
INPUT_FILE = 'data/us_equities_data.csv'
OUTPUT_SEQ_FILE = 'data/sequences.npy'
OUTPUT_LABELS_FILE = 'data/labels.npy'
OUTPUT_METADATA_FILE = 'data/metadata.npy'
OUTPUT_CSV_FILE = 'log/TrainingLog.csv'
MAX_WORKERS = min(11, os.cpu_count() - 1)

def calculate_price_distance(df, column):
    """Calculate percentage distance of closing price to a moving average."""
    return (df['Close'] / df[column] - 1) * 100

def count_rsl_nh(df):
    """Count cumulative RSL_NH occurrences."""
    return df['RSL_NH'].cumsum()

def calculate_rsl_slope(df, window=10):
    """Calculate RSL slope based on the last 10 days."""
    return df['RSL'].diff(periods=window) / window

def count_up_down_days(df, window=14):
    """Count up days vs down days in closing price over the last 14 days."""
    return df['Close'].diff().rolling(window=window).apply(lambda x: np.sum(x > 0) - np.sum(x < 0))

def process_symbol(symbol):
    try:
        df = fetch_OHLCV(symbol, num_bars=None, interval='D')
        if df is None or df.empty:
            return [], [], []

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

        # Add new features
        df['Distance_to_21EMA'] = calculate_price_distance(df, 'Close_21_bar_ema')
        df['Distance_to_50SMA'] = calculate_price_distance(df, 'Close_50_bar_sma')
        df['Distance_to_200SMA'] = calculate_price_distance(df, 'Close_200_bar_sma')
        df['RSL_NH_Count'] = count_rsl_nh(df)
        df['RSL_Slope'] = calculate_rsl_slope(df)
        df['Up_Down_Days'] = count_up_down_days(df)

        symbol_sequences = []
        symbol_profits = []
        symbol_metadata = []

        for i in range(len(df) - 1):
            # Check if this is the day of a breakout (last day Consol_Detected is true)
            if df['Consol_Detected'].iloc[i] and not df['Consol_Detected'].iloc[i + 1]:
                consol_lhs_price = df['Consol_LHS_Price'].iloc[i]
                
                # The sequence ends the day before the breakout
                end_date = df.index[i - 1]
                start_date = df.index[max(0, i - SEQUENCE_LENGTH)]
                
                # Select features for the sequence
                feature_columns = ['Consol_Len_Bars', 'Consol_Depth_Percent',
                                   'Distance_to_21EMA', 'Distance_to_50SMA', 'Distance_to_200SMA', 
                                   'RSL_NH_Count', 'RSL_Slope', 'Up_Down_Days', 
                                   'Stage 2', 'UpDownVolumeRatio', 'ATR', '%B']
                
                seq = df.loc[start_date:end_date, feature_columns].values.tolist()
                if len(seq) < SEQUENCE_LENGTH:
                    padding = [[0] * len(feature_columns)] * (SEQUENCE_LENGTH - len(seq))
                    seq = padding + seq
                
                # Check for the 50-day rule starting from the day after the breakout
                for j in range(i + 1, len(df) - 2):
                    if not (df['Distance_to_50SMA'].iloc[j] < 0 and 
                            df['Distance_to_50SMA'].iloc[j + 1] < 0 and 
                            df['Distance_to_50SMA'].iloc[j + 2] < 0):
                        continue

                    profit = df['Close'].iloc[j + 2] / consol_lhs_price - 1
                    
                    symbol_sequences.append(seq)
                    symbol_profits.append(profit)
                    symbol_metadata.append([symbol, end_date.strftime('%Y-%m-%d')])
                    break

        return symbol_sequences, symbol_profits, symbol_metadata

    except Exception as e:
        print(f"Error processing symbol {symbol}: {e}")
        return [], [], []

def main():
    print("Loading securities...")
    securities = pd.read_csv(INPUT_FILE)['symbol'].tolist()
    print(f"Total securities: {len(securities)}")

    all_sequences = []
    all_profits = []
    all_metadata = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol, symbol): symbol for symbol in securities}
        
        for future in tqdm(as_completed(futures), total=len(securities), desc="Processing symbols"):
            symbol = futures[future]
            try:
                symbol_sequences, symbol_profits, symbol_metadata = future.result()
                all_sequences.extend(symbol_sequences)
                all_profits.extend(symbol_profits)
                all_metadata.extend(symbol_metadata)
            except Exception as exc:
                print(f'{symbol} generated an exception: {exc}')

    if not all_sequences or not all_profits or not all_metadata:
        raise ValueError("No valid data processed")

    sequences_array = np.array(all_sequences)
    profits_array = np.array(all_profits)
    metadata_array = np.array(all_metadata)

    # Save data
    np.save(OUTPUT_SEQ_FILE, sequences_array)
    np.save(OUTPUT_LABELS_FILE, profits_array)
    np.save(OUTPUT_METADATA_FILE, metadata_array)

    # Save CSV log
    pd.DataFrame(all_metadata, columns=['Symbol', 'Date']).assign(Profit=all_profits).to_csv(OUTPUT_CSV_FILE, index=False)

    print("Data processing completed successfully.")

if __name__ == '__main__':
    main()