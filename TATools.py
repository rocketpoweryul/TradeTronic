# open source modules
import pandas as pd
import numpy as np
from scipy.stats import linregress, rankdata
import tqdm
import concurrent.futures
import time
from numba import jit

# internal modules
from    NorgateInterface import *

def find_swing_high_and_lows(df):
    """
    Detects swing highs and swing lows in a DataFrame containing price data.

    Parameters:
        df (DataFrame): A pandas DataFrame containing 'High' and 'Low' columns representing
                        the high and low prices respectively.

    Returns:
        DataFrame: A copy of the input DataFrame with an additional column 'SwHL' indicating
                   swing highs and lows. 
                   1 for swing high, -1 for swing low, and 0 otherwise.
                   
    Raises:
        ValueError: If 'High' or 'Low' columns are not present in the DataFrame, or if the
                    DataFrame contains fewer than 3 rows of data.
    """
    if 'High' not in df.columns or 'Low' not in df.columns:
        raise ValueError("DataFrame must contain 'High' and 'Low' columns")
    
    if df.shape[0] < 3:
        raise ValueError("DataFrame must contain at least 3 rows of data")
    
    #df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    
    df['SwHL'] = 0

    for i in range(1, len(df) - 1):
        # determine current high and low values
        curr_high = df['High'].iloc[i    ]
        next_high = df['High'].iloc[i + 1]
        prev_high = df['High'].iloc[i - 1]
        
        curr_low = df['Low'].iloc[i    ]
        next_low = df['Low'].iloc[i + 1]
        prev_low = df['Low'].iloc[i - 1]

        # detect swing high
        is_swing_high = curr_high > next_high and curr_high > prev_high

        # detect swing low
        is_swing_low = curr_low < next_low and curr_low < prev_low

        # add detection results to dataframe
        if is_swing_high:
            df.at[df.index[i], 'SwHL'] = 1
        elif is_swing_low:
            df.at[df.index[i], 'SwHL'] = -1

    return df

def filter_swing_high_and_lows(df):
    """
    Filter swing high and low points in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing columns 'SwHL', 'High', and 'Low'.

    Returns:
        pandas.DataFrame: DataFrame with swing high and low points filtered.

    Raises:
        ValueError: If DataFrame does not contain 'SwHL', 'High', and 'Low' columns,
                    or if DataFrame contains fewer than 3 data points.
    """
    if 'SwHL' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
        raise ValueError("DataFrame must contain 'SwHL', 'High', and 'Low' columns.")
    if len(df) < 3:
        raise ValueError("DataFrame must contain at least 3 data points.")
    
    # Create a copy of the SwHL column to modify it
    filtered_SwHL = df['SwHL'].copy()
    
    # Variables to hold the current retained high/low and their indices
    current_high = None
    current_high_index = None
    current_low = None
    current_low_index = None
    
    # Loop through the DataFrame using iloc for positional indexing
    for i in range(1, len(df)):
        if df['SwHL'].iloc[i] == 1:  # Current point is a swing high
            if current_high is None:
                # First swing high found
                current_high = df['High'].iloc[i]
                current_high_index = i
                # reset swing lows since streak is killed
                current_low = None
                current_low_index = None
            else:
                # Check if this high is better than the last retained high
                if df['High'].iloc[i] > current_high:
                    # New higher swing high, replace the previous
                    filtered_SwHL.iloc[current_high_index] = 0
                    current_high = df['High'].iloc[i]
                    current_high_index = i
                else:
                    # Lower swing high, filter it out
                    filtered_SwHL.iloc[i] = 0

        elif df['SwHL'].iloc[i] == -1:  # Current point is a swing low
            if current_low is None:
                # First swing low found
                current_low = df['Low'].iloc[i]
                current_low_index = i
                # reset swing highs since streak is killed
                current_high = None
                current_high_index = None
            else:
                # Check if this low is better than the last retained low
                if df['Low'].iloc[i] < current_low:
                    # New lower swing low, replace the previous
                    filtered_SwHL.iloc[current_low_index] = 0
                    current_low = df['Low'].iloc[i]
                    current_low_index = i
                else:
                    # Higher swing low, filter it out
                    filtered_SwHL.iloc[i] = 0

    # Replace original SwHL with filtered values
    df['SwHL'] = filtered_SwHL
    return df

def filter_peaks(df):
    """
    Identify and filter peak points in a stock price DataFrame.

    This function analyzes a DataFrame containing stock price information and identifies the peak points.
    A peak point is defined as a swing high that is higher than the preceding and following swing highs,
    or a swing low that is lower than the preceding and following swing lows.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing at least 'SwHL', 'High', and 'Low' columns.
        'SwHL' column should have values -1, 0, or 1 indicating swing lows, no swings, and swing highs respectively.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'Peak' column. The 'Peak' column contains 
        1 for identified peak swing highs,
        -1 for identified peak swing lows, 
        and 0 otherwise.

    Raises:
        ValueError: If the input DataFrame does not contain 'SwHL', 'High', and 'Low' columns.
    """
    
    if not {'High', "Low", "SwHL"}.issubset(df.columns):
        raise ValueError("Dataframe must contain 'High', 'Low', 'SwHL' columns.")
    
    # init a new peak column with zeros
    df['Peak'] = 0
    
    # get indices for swing highs and lows
    swing_highs = df.index[df['SwHL'] ==  1].tolist()
    swing_lows  = df.index[df['SwHL'] == -1].tolist()
    
    # loop through swing highs, starting at the second occurrence and ending at 2nd last occurrence given the rules
    for i in range( 1, len(swing_highs) - 1 ):
        prev_swing_high_price = df.loc[swing_highs[i-1], 'High']
        curr_swing_high_price = df.loc[swing_highs[i  ], 'High']
        next_swing_high_price = df.loc[swing_highs[i+1], 'High']
        
        # Find the index of the next swing low after the current swing high
        next_swing_low_indices = [low for low in swing_lows if low > swing_highs[i]]
        next_swing_low_price = df.loc[next_swing_low_indices[0], 'Low'] if next_swing_low_indices else None
        
        # detect positive peak
         # Detect positive peak
        if curr_swing_high_price > prev_swing_high_price and \
          (curr_swing_high_price > next_swing_high_price or  \
           (next_swing_low_price is not None and prev_swing_high_price > next_swing_low_price)):
            df.loc[swing_highs[i], 'Peak'] = 1
            
    # loop through swing lows, starting at the second occurrence and ending at 2nd last occurrence given the rules
    for i in range( 1, len(swing_lows) - 1 ):
        prev_swing_low_price  = df.loc[swing_lows[ i-1], 'Low' ]
        curr_swing_low_price  = df.loc[swing_lows[ i  ], 'Low' ]
        next_swing_low_price  = df.loc[swing_lows[ i+1], 'Low' ]
        
        # find the index of the next swing high after the current swing low
        next_swing_high_indices = [high for high in swing_highs if high > swing_lows[i]]
        next_swing_high_price = df.loc[next_swing_high_indices[0], 'High'] if next_swing_high_indices else None

        # Detect negative peak
        if curr_swing_low_price < prev_swing_low_price and \
           (curr_swing_low_price < next_swing_low_price or \
            (next_swing_high_price is not None and prev_swing_low_price < next_swing_high_price)):
            df.loc[swing_lows[i], 'Peak'] = -1
            
    return df    

def detect_consolidation(df, consol_min_bars=20, consol_mindepth_pct=5, consol_maxdepth_pct=35):
    """
    Analyze the entire stock history to detect consolidation phases and append the DataFrame with new columns.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing 'High', 'Low', 'SwHL', and 'Peak' columns.
        consol_min_bars (int): The minimum number of bars to consider a period as consolidation.
        consol_mindepth_pct (float): The minimum depth percentage to consider for consolidation.
        consol_maxdepth_pct (float): The maximum depth percentage to consider for consolidation.

    Returns:
        pandas.DataFrame: The input DataFrame appended with new columns indicating consolidation status and metrics:
        - Consol_Detected
        - Consol_LHS_Price
        - Consol_Len_Bars
        - Consol_Depth_Percent
    """
    
    # Create output dataframe columns with default values.
    df['Consol_Detected'        ] = False
    df['Consol_LHS_Price'       ] = 0.0
    df['Consol_Len_Bars'        ] = 0
    df['Consol_Depth_Percent'   ] = 0.0
    
    candidate_Consol_Detected      = False
    candidate_Consol_LHS_Price     = 0.0
    candidate_Consol_Depth_Percent = 0.0
    candidate_Consol_Lowest_Price  = 0.0
    
    for index, row in df.iterrows():
        # Get the position of the index in DataFrame's index
        index_pos = df.index.get_loc(index)  

        # get current row high and low
        row_high = df.at[index, 'High']
        row_low  = df.at[index, 'Low' ]

        #### CONFIRM START OF CONSOLIDATION
        if not candidate_Consol_Detected:
            # If Consol_Detected for the previous row is false, then we are still searching for the next consolidation
            # this is done by inspecting peaks previous to the current row

            # Create List of Filtered Peaks
            peaks = df[(df['Peak'] == 1) & (df.index < index) & (df['High'] > row['High']) ]

            # Skip loop iteration if there are no peaks
            if peaks.empty:
                continue

            # Create the list for peak indices, filtered per above
            peak_indices = peaks.index.tolist()

            # Search for a candidate LHS to the consolidation
            old_peak_price = 0
            for peak_index in reversed(peak_indices):
                # get distance of index from peak_index
                bars_since_peak = df.index.get_loc(index) - df.index.get_loc(peak_index)
                
                # get high price of peak
                peak_high_price = df.at[peak_index, 'High']
                
                # get low price between peak_index and index
                filtered_df = df.loc[peak_index:index]
                lowest_price_since_peak = filtered_df['Low'].min()
                drawdown_since_peak = 100*(peak_high_price - lowest_price_since_peak)/peak_high_price
                
                # check is peak is valid to start the consolidation
                if (bars_since_peak >= consol_min_bars) and (drawdown_since_peak > consol_mindepth_pct) and (drawdown_since_peak < consol_maxdepth_pct) and (row_high < peak_high_price) and (peak_high_price > old_peak_price):
                    candidate_Consol_Detected      = True
                    candidate_Consol_LHS_Price     = peak_high_price
                    candidate_Consol_Depth_Percent = drawdown_since_peak
                    candidate_Consol_Lowest_Price  = lowest_price_since_peak
                    candidate_Consol_Peak_Index    = peak_index

                message = ''
                if   bars_since_peak < consol_min_bars:
                    message = f"Base is too short at {bars_since_peak} bars, "
                if drawdown_since_peak < consol_mindepth_pct:
                    message += f"Drawdown is too shallow at {drawdown_since_peak:.2F}, "
                if drawdown_since_peak > consol_maxdepth_pct:
                    message += f"Drawdown is too deep at {drawdown_since_peak:.2F}."

                if old_peak_price < peak_high_price:
                    old_peak_price = peak_high_price
        
        ### Confirm still in base
        else:
            # It's the last bar so if the base is valid, we need to confirm the consolidation if it exists regardless.
            if index == df.index[-1]:
                # Calculate the integer positions of the start and end indices
                start_pos = df.index.get_loc(candidate_Consol_Peak_Index) + 1  # position right after peak_index
                end_pos = df.index.get_loc(old_row_index)     # position of old_row_index

                # Set consolidation data in the dataframe from the row after the peak to the old row index
                num_rows = end_pos - start_pos + 1
                
                df.iloc[start_pos:end_pos + 1, df.columns.get_loc('Consol_Detected')] = True
                df.iloc[start_pos:end_pos + 1, df.columns.get_loc('Consol_LHS_Price')] = candidate_Consol_LHS_Price
                df.iloc[start_pos:end_pos + 1, df.columns.get_loc('Consol_Len_Bars')] = list(range(num_rows))  
                df.iloc[start_pos:end_pos + 1, df.columns.get_loc('Consol_Depth_Percent')] = candidate_Consol_Depth_Percent

            # Price breaks out of consolidation, base was valid but needs to be reset
            if row_high > candidate_Consol_LHS_Price:
                # Calculate the integer positions of the start and end indices
                start_pos = df.index.get_loc(candidate_Consol_Peak_Index) + 1  # position right after peak_index
                end_pos = index_pos - 1     # position of old_row_index

                # Set consolidation data in the dataframe from the row after the peak to the old row index
                num_rows = end_pos - start_pos + 1
                
                df.iloc[start_pos:end_pos + 1, df.columns.get_loc('Consol_Detected')] = True
                df.iloc[start_pos:end_pos + 1, df.columns.get_loc('Consol_LHS_Price')] = candidate_Consol_LHS_Price
                df.iloc[start_pos:end_pos + 1, df.columns.get_loc('Consol_Len_Bars')] = list(range(num_rows))  
                df.iloc[start_pos:end_pos + 1, df.columns.get_loc('Consol_Depth_Percent')] = candidate_Consol_Depth_Percent

                # reset candidate consolidation data so loop checks for next consolidation
                candidate_Consol_Detected      = False
                candidate_Consol_LHS_Price     = 0.0
                candidate_Consol_Depth_Percent = 0.0
                candidate_Consol_Lowest_Price  = 0.0
            
            # now check if the low of base is undercut and if this makes the base too deep
            elif row_low < candidate_Consol_Lowest_Price:
                drawdown_since_peak = 100*(peak_high_price - row_low)/peak_high_price
                
                # if the new drawdown is indeed to deep, cancel the base detection
                if drawdown_since_peak > consol_maxdepth_pct:
                    # reset candidate consolidation data so loop checks for next consolidation
                    candidate_Consol_Detected      = False
                    candidate_Consol_LHS_Price     = 0.0
                    candidate_Consol_Depth_Percent = 0.0
                    candidate_Consol_Lowest_Price  = 0.0
                    drawdown_since_peak            = 0.0

                else:
                    candidate_Consol_Depth_Percent = 100*(candidate_Consol_LHS_Price - row_low)/candidate_Consol_LHS_Price
                    candidate_Consol_Lowest_Price  = row_low
            
        # otherwise the base is just continuing. 
        # 
        # If this is the last index, also copy the previous values to be consistent
        if index == df.index[-1] and df.loc[old_row_index, 'Consol_Detected']:
            df.loc[index, 'Consol_Detected'      ] = df.loc[old_row_index, 'Consol_Detected'      ]
            df.loc[index, 'Consol_LHS_Price'     ] = df.loc[old_row_index, 'Consol_LHS_Price'     ]
            df.loc[index, 'Consol_Len_Bars'      ] = df.loc[old_row_index, 'Consol_Len_Bars'      ] + 1
            df.loc[index, 'Consol_Depth_Percent' ] = df.loc[old_row_index, 'Consol_Depth_Percent' ]
        
        # log the old row index for use when setting dataframe per above.
        old_row_index = index
    return df

def add_moving_average(df, num_bars=21, MA_type='EMA', price='Close'):
    """
    Adds a moving average to a pandas DataFrame based on user-specified parameters.

    Parameters:
        df (pandas.DataFrame): A pandas DataFrame containing at least one column
            representing the stock's price.
        num_bars (int, optional): The number of bars (or rows in the dataframe)
            to use for calculating the moving average. Defaults to 21.
        MA_type (str, optional): The type of moving average to calculate. Can be
            either 'SMA' for simple moving average or 'EMA' for exponential
            moving average. Defaults to 'EMA'.
        price (str, optional): The column name in the dataframe that represents
            the stock's price.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional column containing
            the calculated moving average.

    Raises:
        ValueError: If 
            - The input DataFrame is empty 
            - The specified 'price' column does not exist in the DataFrame.
            - The MA type is not supported
    """
    
    # Check if the dataframe is not empty
    if df.empty:
        raise ValueError("The input dataframe is empty.")
        
    # Check if the 'price' column exists in the dataframe
    if price not in df.columns:
        raise ValueError(f"The column '{price}' does not exist in the dataframe.")

    # Calculate the moving average based on the MA_type
    if MA_type.lower() == 'sma':
        # Calculate simple moving average using rolling mean
        ma = df[price].rolling(window=num_bars).mean()
    elif MA_type.lower() == 'ema':
        # Calculate exponential moving average using ewm
        ma = df[price].ewm(span=num_bars, adjust=False).mean()
    else:
        raise ValueError(f"Invalid MA_type. It should be either 'SMA' or 'EMA'.")

    # Create the new column name
    ma_name = f"{price}_{num_bars}_bar_{MA_type}"
    
    # Add the moving average as a new column to the dataframe
    df[ma_name] = ma

    return df

def add_relative_strength_line(df, index="S&P500", new_high_bars=69):
    """
    Adds a relative strength line (RSL) and a new high column to a pandas DataFrame.

    Parameters:
        df (pandas.DataFrame): A pandas DataFrame containing at least a 'Close' column
            with stock prices and a DatetimeIndex.
        index (str, optional): The market index to use for calculating relative strength.
            Can be 'S&P500', 'NASDAQ', or 'DJIA'. Defaults to 'S&P500'.
        new_high_bars (int, optional): The number of bars to use for calculating new highs
            in the RSL. Defaults to 69.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns for the relative
            strength line and new highs.

    Raises:
        ValueError: If
            - The input DataFrame is empty.
            - The input DataFrame does not have a DatetimeIndex.
            - The input DataFrame does not have a 'Close' column.
            - The specified index is not supported.
            - The frequency of the DataFrame index is not daily, weekly, or monthly.
    """
    
    # Check if the dataframe is not empty
    if df.empty:
        raise ValueError("The input dataframe is empty.")
    
    # Check if the dataframe has a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index of df must be a pandas DatetimeIndex.")
    
    # Check if the 'Close' column exists in the dataframe
    if 'Close' not in df.columns:
        raise ValueError("The dataframe must contain a 'Close' column.")
    
    # Determine period of input stock data in df
    date_diff = df.index.to_series().diff().dt.days
    min_date_diff = date_diff.min()
    
    # Map minimum date difference to interval
    if min_date_diff == 1:
        interval = 'D'
    elif min_date_diff == 7:
        interval = 'W'
    elif min_date_diff >= 30:
        interval = 'M'
    else:
        raise ValueError("Invalid frequency: {}. Please choose daily (1 day), weekly (7 days), or monthly (30+ days).".format(min_date_diff))
    
    # Get the index symbol based on the input
    index_symbol_map = {"S&P500": "$SPX", "NASDAQ": "$COMP", "DJIA": "$DJI"}
    index_symbol = index_symbol_map.get(index)
    if index_symbol is None:
        raise ValueError("Invalid index specified: '{}'. Please choose from S&P500, NASDAQ, or DJIA.".format(index))
    
    # Fetch OHLCV data for the index
    index_df = fetch_OHLCV(index_symbol, len(df), interval=interval)
    
    # Check if the fetched index dataframe is not empty and contains a 'Close' column
    if index_df.empty or 'Close' not in index_df.columns:
        raise ValueError(f"Failed to fetch valid OHLCV data for index '{index}'.")
    
    # Update the dataframe with RSL and RSL new high data
    df['RSL'] = df['Close'] / index_df['Close']
    # Calculate new highs over the specified period using a rolling window
    df['RSL_NH'] = df['RSL'] > df['RSL'].rolling(window=new_high_bars, min_periods=new_high_bars).max().shift(1)
    
    return df

@jit(nopython=True)
def compute_slopes(close_prices, num_bars):
    total_bars = len(close_prices)
    num_bars_to_use = min(num_bars, total_bars)

    x = np.arange(num_bars_to_use)
    x_mean = np.mean(x)
    slopes = np.empty(total_bars - num_bars_to_use + 1)

    for i in range(num_bars_to_use, total_bars + 1):
        y = close_prices[i - num_bars_to_use:i]
        if np.any(np.isnan(y)):  # Check for missing values
            slopes[i - num_bars_to_use] = np.nan
        else:
            y_mean = np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            slopes[i - num_bars_to_use] = slope

    return slopes

def compute_slope_for_symbol(symbol, num_bars):
    try:
        # Fetch data
        start_fetch = time.time()
        df = fetch_OHLCV(symbol=symbol, num_bars=num_bars*3, interval='D')
        fetch_duration = time.time() - start_fetch

        # Check if data fetching was successful
        if df.empty or len(df) < 2:
            print(f"\nWarning: Insufficient data for {symbol}")
            return symbol, None, fetch_duration, 0

        # Compute slopes
        start_regression = time.time()
        adjusted_num_bars = min(num_bars, len(df))
        slopes = compute_slopes(df['Close'].values, adjusted_num_bars)
        slope_series = pd.Series(slopes, index=df.index[adjusted_num_bars-1:])
        regression_duration = time.time() - start_regression

        return symbol, slope_series, fetch_duration, regression_duration
    
    except ZeroDivisionError:
        print(f"Error processing symbol {symbol}: division by zero")
        return symbol, None, 0, 0
    
    except ValueError as ve:
        print(f"Error processing symbol {symbol}: {ve}")
        return symbol, None, 0, 0
    
    except Exception as e:
        print(f"Error processing symbol {symbol}: {e}")
        return symbol, None, 0, 0

def Compute_Rel_Strength_LR(symbol_list, num_bars=252):
    slope_dict = {}
    fetch_times = []
    regression_times = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(compute_slope_for_symbol, symbol, num_bars): symbol for symbol in symbol_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbol_list), desc='Processing symbols'):
            symbol, slope_series, fetch_duration, regression_duration = future.result()
            fetch_times.append(fetch_duration)
            regression_times.append(regression_duration)
            if slope_series is not None:
                slope_dict[symbol] = slope_series

    if not slope_dict:
        print("Error: No slopes calculated for any symbol")
        return None

    # Save the slope_dict to CSV
    slope_df = pd.concat(slope_dict.values(), axis=1, keys=slope_dict.keys())
    slope_df.to_csv('slope_dict.csv', index=True)

    combined_df = pd.concat(slope_dict.values(), axis=1, keys=slope_dict.keys())

    # Using the dates directly from the combined DataFrame
    combined_df = combined_df.reindex(combined_df.index)

    valid_stocks_count = combined_df.notna().sum(axis=1)
    ranks_df = combined_df.rank(axis=1, method='min', na_option='bottom')
    ranks_df = ranks_df.div(valid_stocks_count, axis=0).multiply(100).clip(0, 100)

    # Convert to integers, ensuring we handle non-finite values properly
    percentile_ranks_df = ranks_df.apply(lambda row: row.map(lambda v: int(v) if pd.notna(v) else -1), axis=1)

    print(f"Average time for fetching data: {np.mean(fetch_times):.4f} seconds")
    print(f"Average time for regression calculations: {np.mean(regression_times):.4f} seconds")

    return percentile_ranks_df

def update_stock_dataframe_with_rs(df, RS_LR, window=63):
    """
    Updates the stock DataFrame by appending RS values and a column indicating new highs in RS.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the stock's historical data.
        RS_LR (pandas.DataFrame): DataFrame containing the relative strength ranks for all stocks.
        window (int, optional): The rolling window size to check for new RS highs. Defaults to 63 (approx. 3 months).

    Returns:
        pandas.DataFrame: Updated DataFrame with RS values and new high indication.
    """
    # Ensure the DateString column is in datetime format for proper alignment
    df['DateString'] = pd.to_datetime(df['DateString'])
    RS_LR.index = pd.to_datetime(RS_LR.index)
    
    # Extract the ticker symbol from the stock DataFrame
    ticker = df['Symbol'].iloc[0]  # Assumes the ticker symbol is in the 'Symbol' column

    if ticker not in RS_LR.columns:
        raise ValueError(f"Ticker {ticker} not found in RS_LR DataFrame.")
    
    # Align the RS values with the stock DataFrame using DateString
    df.set_index('DateString', inplace=True)
    df['RS'] = RS_LR[ticker].reindex(df.index)

    # Calculate the rolling maximum for the RS values
    df['RS_New_High'] = df['RS'] == df['RS'].rolling(window, min_periods=1).max()
    
    # Reset the index to its original integer-based index
    df.reset_index(inplace=True)

    return df

def get_stage2_uptrend(df):
    """
    This function determines if the stock represented by the dataframe is in a stage 2 uptrend.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the stock's historical data.
        The required columns are 'Close', '50_bar_sma', '150_bar_sma', '200_bar_sma', 'RS'.

    Returns:
    pandas.Series: A boolean Series indicating whether the stock is in a stage 2 uptrend.
    """
    # Check if all required columns exist
    required_columns = ['Close', 'Close_50_bar_sma', 'Close_150_bar_sma', 'Close_200_bar_sma', 'RS']
    if not all(column in df.columns for column in required_columns):
        missing_cols = set(required_columns) - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Calculate the 52 week low and high
    df['52_week_low'] = df['Close'].rolling(window=252).min()
    df['52_week_high'] = df['Close'].rolling(window=252).max()

    # Define the stage 2 uptrend rules
    rules = [
        (df['Close_150_bar_sma'] > df['Close_200_bar_sma']),  # 1. 150 day SMA is above 200 day SMA
        (df['Close_50_bar_sma'] > df['Close_150_bar_sma']),  # 2. 50 day SMA is above the 150 day SMA
        (df['Close'] > df['Close_150_bar_sma']),  # 3. Closing price is above the 150 day SMA
        (df['Close_200_bar_sma'].diff(21) > 0),  # 4. 200-day SMA is uptrending for at least one month (21 days)
        ((df['Close'] - df['52_week_low']) / df['52_week_low'] > 0.25),  # 5. Closing price is at least 25% above the 52 week low
        ((df['Close'] / df['52_week_high']) > 0.75),  # 6. Closing price is within 25% of the 52 week high
        (df['RS'] > 70)  # 7. Relative Strength (RS) is greater than 70
    ]

    # Combine all rules using logical AND operation
    df['Stage 2'] = np.all(rules, axis=0)

    # Identify which rule(s) are failing
    for i, rule in enumerate(rules):
        if not rule.iloc[-1]:  # Check the last row of the dataframe
            print(f"Rule {i+1} failed: {rule.name}")
    
    return df
    