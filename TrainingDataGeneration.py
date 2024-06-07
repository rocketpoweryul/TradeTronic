import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import multiprocessing

# functions from this project
from NorgateInterface import *
from TATools import *

# hyperparameters
sequence_length = 252
relevant_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover', 'Consol_Detected',
    'Consol_Len_Bars', 'Consol_Depth_Percent', 'Close_21_bar_ema',
    'Close_50_bar_sma', 'Close_150_bar_sma', 'Close_200_bar_sma',
    'RSL', 'RSL_NH'
]

def process_symbol(symbol):
    try:
        # Get stock data for processing
        df = fetch_OHLCV(symbol, num_bars=None, interval='D')

        # Initial stock dataframe processing
        df = find_swing_high_and_lows(df)
        df = filter_swing_high_and_lows(df)
        df = filter_peaks(df)
        df = detect_consolidation(df)
        df = add_moving_average(df, 21, 'ema')
        df = add_moving_average(df, 50, 'sma')
        df = add_moving_average(df, 150, 'sma')
        df = add_moving_average(df, 200, 'sma')
        df = add_relative_strength_line(df)

        # Convert index to integers, but save dates as datestrings 
        df['DateString'] = df.index.strftime('%Y-%m-%d')  # Save date info before resetting index

        # Determine Friday dates before resetting the index for plotting purposes
        df['DayOfWeek'] = df.index.weekday  # Monday=0, Sunday=6

        # Reset index to use numerical index
        df.reset_index(drop=True, inplace=True)

        symbol_sequences = []
        symbol_profits = []
        symbol_metadata = []

        for index in range(len(df)):
            if df['Consol_Detected'][index] == True:
                breakout = False
                consol_lhs_price = df['Consol_LHS_Price'][index]
                
                if index + 1 < len(df):
                    if df['Consol_Detected'][index + 1] == False:
                        breakout = True
                        start_index = max(0, index - sequence_length + 1)
                        
                        # Create the sequence of data for this breakout
                        seq = df[relevant_columns].iloc[start_index:index + 1].values.tolist()

                        # Pad sequences if they are shorter than sequence_length
                        if len(seq) < sequence_length:
                            padding = [[0] * len(relevant_columns)] * (sequence_length - len(seq))
                            seq = padding + seq
                        
                        # Find the next occurrence where Close price is below Close_50_bar_sma for three consecutive bars
                        for i in range(index + 1, len(df) - 2):
                            if (df['Close'][i] < df['Close_50_bar_sma'][i] and 
                                df['Close'][i + 1] < df['Close_50_bar_sma'][i + 1] and 
                                df['Close'][i + 2] < df['Close_50_bar_sma'][i + 2]):
                                
                                profit = df['Close'][i + 2] / consol_lhs_price - 1
                                
                                # Append valid sequence and profit
                                symbol_sequences.append(seq)
                                symbol_profits.append(profit)
                                
                                # Append metadata to metadata list with correct breakout date
                                meta = [symbol, df['DateString'][index + 1]]  # Adjust index to get the correct date
                                symbol_metadata.append(meta)
                                break  # Exit the loop after finding the valid profit

        return symbol_sequences, symbol_profits, symbol_metadata

    except Exception as e:
        print(f"Error processing symbol {symbol}: {e}")
        return [], [], []

def load_sec_list(filename):
    sec_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # Skip the header row
        next(reader)
        sec_list = [row[0] for row in reader]
    return sec_list

if __name__ == '__main__':
    # Get the list of securities
    print("Accessing list of securities...")
    try:
        print("Loading us_equities_data.csv file")
        sec_list = load_sec_list('us_equities_data.csv')

        if not sec_list:
            raise ValueError("No securities found in the input file")

        print(f"Total securities: {len(sec_list)}")

        # Initialize training data
        all_sequences = []  # Captures x for the training data
        all_profits = []    # Captures y for the training data
        all_metadata = []   # Captures metadata for each identified training example

        # Process symbols in parallel
        num_workers = min(14, os.cpu_count() - 2)  # Adjust based on available CPUs
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in sec_list}
            for future in tqdm(as_completed(future_to_symbol), total=len(sec_list), desc="Processing symbols"):
                symbol = future_to_symbol[future]
                try:
                    symbol_sequences, symbol_profits, symbol_metadata = future.result()
                    all_sequences.extend(symbol_sequences)
                    all_profits.extend(symbol_profits)
                    all_metadata.extend(symbol_metadata)
                except Exception as exc:
                    print(f'{symbol} generated an exception: {exc}')

        if not all_sequences or not all_profits or not all_metadata:
            raise ValueError("No valid data processed")

        # Convert sequences and profits to numpy arrays for normalization
        sequences_array = np.array(all_sequences)
        profits_array = np.array(all_profits).reshape(-1, 1)

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Fit and transform the sequences array
        sequences_shape = sequences_array.shape
        sequences_array = scaler.fit_transform(sequences_array.reshape(-1, sequences_array.shape[-1])).reshape(sequences_shape)

        # Directly use raw profit values for CSV output
        raw_profits = profits_array.flatten().tolist()

        # Save sequences
        np.save('sequences.npy', sequences_array)

        # Save labels (raw profits)
        labels = profits_array.flatten().tolist()
        np.save('labels.npy', labels)

        # Save metadata
        np.save('metadata.npy', all_metadata)

        # Combine metadata with raw profits for CSV output
        combined_data = [[meta[0], meta[1], profit] for meta, profit in zip(all_metadata, labels)]

        # Save csv record
        with open('TrainingLog.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Symbol', 'Date', 'Profit'])
            writer.writerows(combined_data)
    except Exception as e:
        print(f"Error in main execution: {e}")
