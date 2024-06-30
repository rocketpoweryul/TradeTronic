# open source modules
import pandas as pd
import numpy as np
import tqdm
import concurrent.futures
from scipy.stats import rankdata

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

    df['SwHL'] = 0
    df['prev_High'] = df['High'].shift(1)
    df['next_High'] = df['High'].shift(-1)
    df['prev_Low'] = df['Low'].shift(1)
    df['next_Low'] = df['Low'].shift(-1)

    df['SwHL'] = np.where((df['High'] > df['prev_High']) & (df['High'] > df['next_High']), 1, df['SwHL'])
    df['SwHL'] = np.where((df['Low'] < df['prev_Low']) & (df['Low'] < df['next_Low']), -1, df['SwHL'])

    df.drop(['prev_High', 'next_High', 'prev_Low', 'next_Low'], axis=1, inplace=True)
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
    if not {'SwHL', 'High', 'Low'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'SwHL', 'High', and 'Low' columns.")
    if len(df) < 3:
        raise ValueError("DataFrame must contain at least 3 data points.")
    
    # Convert columns to numpy arrays for faster access
    SwHL = df['SwHL'].values
    High = df['High'].values
    Low = df['Low'].values

    # Variables to hold the current retained high/low and their indices
    current_high = None
    current_high_index = None
    current_low = None
    current_low_index = None

    # Iterate through the DataFrame once
    for i in range(1, len(df)):
        if SwHL[i] == 1:  # Current point is a swing high
            if current_high is None or High[i] > current_high:
                # Found a new higher swing high, update current_high
                if current_high is not None:
                    SwHL[current_high_index] = 0  # Reset the previous swing high
                current_high = High[i]
                current_high_index = i
                # Reset swing lows since the streak is killed
                current_low = None
                current_low_index = None
            else:
                SwHL[i] = 0  # Lower swing high, filter it out

        elif SwHL[i] == -1:  # Current point is a swing low
            if current_low is None or Low[i] < current_low:
                # Found a new lower swing low, update current_low
                if current_low is not None:
                    SwHL[current_low_index] = 0  # Reset the previous swing low
                current_low = Low[i]
                current_low_index = i
                # Reset swing highs since the streak is killed
                current_high = None
                current_high_index = None
            else:
                SwHL[i] = 0  # Higher swing low, filter it out

    # Replace original SwHL with filtered values
    df['SwHL'] = SwHL
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
    
    if not {'High', 'Low', 'SwHL'}.issubset(df.columns):
        raise ValueError("Dataframe must contain 'High', 'Low', 'SwHL' columns.")
    
    # Initialize a new Peak column with zeros
    df['Peak'] = 0
    
    # Convert columns to numpy arrays for faster access
    SwHL = df['SwHL'].values
    High = df['High'].values
    Low = df['Low'].values
    Peak = df['Peak'].values
    
    # Get indices for swing highs and lows
    swing_highs = [i for i in range(len(SwHL)) if SwHL[i] == 1]
    swing_lows = [i for i in range(len(SwHL)) if SwHL[i] == -1]
    
    # Loop through swing highs, starting at the second occurrence and ending at the second last occurrence
    for i in range(1, len(swing_highs) - 1):
        prev_swing_high_price = High[swing_highs[i-1]]
        curr_swing_high_price = High[swing_highs[i]]
        next_swing_high_price = High[swing_highs[i+1]]
        
        # Find the index of the next swing low after the current swing high
        next_swing_low_indices = [low for low in swing_lows if low > swing_highs[i]]
        next_swing_low_price = Low[next_swing_low_indices[0]] if next_swing_low_indices else None
        
        # Detect positive peak
        if curr_swing_high_price > prev_swing_high_price and \
          (curr_swing_high_price > next_swing_high_price or \
           (next_swing_low_price is not None and prev_swing_high_price > next_swing_low_price)):
            Peak[swing_highs[i]] = 1
    
    # Loop through swing lows, starting at the second occurrence and ending at the second last occurrence
    for i in range(1, len(swing_lows) - 1):
        prev_swing_low_price = Low[swing_lows[i-1]]
        curr_swing_low_price = Low[swing_lows[i]]
        next_swing_low_price = Low[swing_lows[i+1]]
        
        # Find the index of the next swing high after the current swing low
        next_swing_high_indices = [high for high in swing_highs if high > swing_lows[i]]
        next_swing_high_price = High[next_swing_high_indices[0]] if next_swing_high_indices else None
        
        # Detect negative peak
        if curr_swing_low_price < prev_swing_low_price and \
           (curr_swing_low_price < next_swing_low_price or \
            (next_swing_high_price is not None and prev_swing_low_price < next_swing_high_price)):
            Peak[swing_lows[i]] = -1
    
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

def compute_slope(y):
    """
    Computes the slope of the best-fit line for the given data points in a pandas Series.

    Parameters:
    y (pandas Series): A Series of numerical values representing the dependent variable.

    Returns:
    float: The slope of the best-fit line.

    Raises:
    ValueError: If the input is not a pandas Series, or if the length of y is less than 2.
    """
    
    # Ensure the input is a pandas Series
    if not isinstance(y, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    
    # Check if the length of the Series is sufficient to compute a slope
    if len(y) < 2:
        raise ValueError("At least two data points are required to compute a slope.")
    
    n = len(y)
    x = np.arange(n)  # Generate a sequence of integers [0, 1, ..., n-1]
    
    # Calculate the necessary sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x ** 2)
    sum_xy = np.sum(x * y)
    
    # Compute the slope using the formula for the least squares method
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    
    return slope

def process_symbol(symbol, num_bars):
    """
    Processes the OHLCV data for a given symbol to compute the slopes of closing prices over sliding windows.

    Parameters:
    symbol (str): The stock symbol for which the OHLCV data is to be fetched.
    num_bars (int): The number of bars (periods) to be used in each sliding window for slope computation.

    Returns:
    tuple: A tuple containing the symbol and a DataFrame of slopes indexed by date. If the data is insufficient or missing, returns (symbol, None).

    Raises:
    ValueError: If `num_bars` is not a positive integer.
    """
    
    # Validate num_bars input
    if not isinstance(num_bars, int) or num_bars <= 0:
        raise ValueError("num_bars must be a positive integer.")
    
    # Fetch OHLCV data
    data = fetch_OHLCV(symbol=symbol, num_bars=num_bars*3, interval='D')

    # Ensure there is enough data and that 'Close' column exists
    if data is None or len(data) < num_bars or 'Close' not in data.columns:
        return symbol, None

    # Ensure the index is datetime and sorted
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Compute slopes for each window of 'num_bars'
    slopes = [np.nan] * (num_bars - 1)  # Initialize with NaNs for the initial period
    for i in range(num_bars - 1, len(data)):
        window = data['Close'][i - num_bars + 1 : i + 1]  # Use only past data up to the current point
        slope = compute_slope(window)
        slopes.append(slope)

    # Create a DataFrame with the slopes and dates
    slopes_df = pd.DataFrame(slopes, index=data.index, columns=[symbol])

    return symbol, slopes_df

def Compute_Rel_Strength_LR(sec_list, num_bars=69):
    """
    Computes the relative strength using linear regression slopes for a list of securities over a specified number of bars.

    Parameters:
    sec_list (list): A list of stock symbols to be processed.
    num_bars (int): The number of bars (periods) to be used in each sliding window for slope computation. Default is 69.

    Returns:
    pd.DataFrame: A DataFrame containing the relative strength linear regression ranks for each stock symbol.

    Raises:
    ValueError: If `sec_list` is not a list or is empty, or if `num_bars` is not a positive integer.
    """

    # Validate input parameters
    if not isinstance(sec_list, list) or not sec_list:
        raise ValueError("sec_list must be a non-empty list of stock symbols.")
    if not isinstance(num_bars, int) or num_bars <= 0:
        raise ValueError("num_bars must be a positive integer.")

    # Initialize dictionary for storing historical slopes from each ticker
    slope_dict = {}

    # Use ProcessPoolExecutor to process symbols in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map each symbol to the process_symbol function
        future_to_symbol = {executor.submit(process_symbol, symbol, num_bars): symbol for symbol in sec_list}
        
        # Use tqdm to display the progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_symbol), total=len(sec_list), desc="Processing stocks"):
            symbol, slopes_df = future.result()
            if slopes_df is not None:
                slope_dict[symbol] = slopes_df

    # Combine all DataFrames horizontally, ensuring no hierarchical columns are created
    combined_df = pd.concat(slope_dict.values(), axis=1)

    # Rename columns to the respective stock symbols
    combined_df.columns = list(slope_dict.keys())

    # Compute row-wise percentile ranks
    RS_LR = combined_df.rank(axis=1, pct=True) * 100 - 1

    # Handle missing values by setting them to -1
    RS_LR = RS_LR.fillna(-1).astype(int)

    return RS_LR

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
        The required columns are 'Close', 'Close_50_bar_sma', 'Close_150_bar_sma', 'Close_200_bar_sma'.

    Returns:
    pandas.DataFrame: The input DataFrame with an additional 'Stage 2' column indicating whether the stock is in a stage 2 uptrend.
    """
    # Check if all required columns exist
    required_columns = ['Close', 'Close_50_bar_sma', 'Close_150_bar_sma', 'Close_200_bar_sma']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Calculate the 52 week low and high
    df['52_week_low'] = df['Close'].rolling(window=252).min()
    df['52_week_high'] = df['Close'].rolling(window=252).max()

    # Define the stage 2 uptrend rules
    rules = pd.DataFrame({
        'Rule1': df['Close_150_bar_sma'] > df['Close_200_bar_sma'],
        'Rule2': df['Close_50_bar_sma'] > df['Close_150_bar_sma'],
        'Rule3': df['Close'] > df['Close_150_bar_sma'],
        'Rule4': df['Close_200_bar_sma'].diff(21) > 0,
        'Rule5': (df['Close'] - df['52_week_low']) / df['52_week_low'] > 0.25,
        'Rule6': df['Close'] / df['52_week_high'] > 0.75
    })

    # Combine all rules using logical AND operation
    df['Stage 2'] = rules.all(axis=1)

    # Optional: Identify which rule(s) are failing (uncomment if needed)
    # failing_rules = ~rules.iloc[-1]
    # if failing_rules.any():
    #     print("Failing rules:", ", ".join(failing_rules[failing_rules].index))

    return df

def calculate_up_down_volume_ratio(df, window=50):
    """
    Calculate the up/down volume ratio for the stock represented by the dataframe.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the stock's historical data.
        The required columns are 'Close' and 'Volume'.
    window (int): The rolling window for calculation (default is 50 days).

    Returns:
    pandas.DataFrame: A DataFrame with a new column 'UpDownVolumeRatio' representing the up/down volume ratio.
    """
    # Check if all required columns exist
    required_columns = ['Close', 'Volume']
    if not all(column in df.columns for column in required_columns):
        missing_cols = set(required_columns) - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Determine up days and down days
    price_change = df['Close'].diff()
    
    # Calculate the total volume on up days and down days over the specified window
    up_volume = df['Volume'].where(price_change > 0, 0)
    down_volume = df['Volume'].where(price_change < 0, 0)
    
    up_volume_sum = up_volume.rolling(window=window).sum()
    down_volume_sum = down_volume.rolling(window=window).sum()
    
    # Calculate ratio, handling division by zero
    df['UpDownVolumeRatio'] = np.where(
        down_volume_sum != 0,
        up_volume_sum / down_volume_sum,
        np.inf  # or you could use np.nan if you prefer
    )
    
    return df

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) for a given stock

    Parameters:
        df (pandas.DataFrame): DataFrame containing the stock's historical data
            The required columns are 'High', 'Low', and 'Close'
        period (int): The period over which to calculate ATR (default is 14)

    Returns:
        pandas.DataFrame: A new DataFrame with an additional column 'ATR' containing the calculated ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    # Add ATR to the dataframe
    df['ATR'] = atr
    
    return df

def calculate_pct_b(df):
    """
    Calculate Bollinger Bands (%B) for a given stock

    Parameters:
        df (pandas.DataFrame): DataFrame containing the stock's historical data
            The required columns are 'Close'

    Returns:
        pandas.DataFrame: A new DataFrame with an additional column '%B' containing the calculated %B values
    """
    close = df['Close']
    window = 20
    
    # Calculate rolling mean and standard deviation
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    
    # Calculate upper and lower Bollinger Bands
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    
    # Calculate %B
    bb_values = (close - lower_band) / (upper_band - lower_band)
    
    # Add %B to the dataframe
    df['%B'] = bb_values
    
    return df

def calculate_williams_r(df, period=14):
    """
    Calculate Williams %R indicator for a given stock

    Parameters:
        df (pandas.DataFrame): DataFrame containing the stock's historical data
            The required columns are 'High', 'Low', and 'Close'
        period (int): The lookback period for calculating Williams %R (default is 14)

    Returns:
        pandas.DataFrame: A new DataFrame with an additional column 'Williams %R' containing the calculated Williams %R values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate highest high and lowest low over the specified period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # Calculate Williams %R
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Add Williams %R to the dataframe
    df['Williams %R'] = williams_r
    
    return df

def calculate_adr(df, period=20):
    """
    Calculate Average Daily Range (ADR) indicator for a given stock

    Parameters:
        df (pandas.DataFrame): DataFrame containing the stock's historical data
            The required columns are 'High' and 'Low'
        period (int): The period for calculating the moving average (default is 20)

    Returns:
        pandas.DataFrame: A new DataFrame with an additional column 'ADR' containing the calculated ADR values
    """
    high = df['High']
    low = df['Low']
    
    # Calculate daily range ratio
    daily_range_ratio = high / low
    
    # Calculate moving average of daily range ratio
    ma_daily_range_ratio = daily_range_ratio.rolling(window=period).mean()
    
    # Calculate ADR
    adr = 100 * (ma_daily_range_ratio - 1)
    
    # Add ADR to the dataframe
    df['ADR'] = adr
    
    return df

def calculate_up_down_ratio(df, period=50):
    """
    Calculate Up/Down Volume Ratio for a given stock

    Parameters:
        df (pandas.DataFrame): DataFrame containing the stock's historical data
            The required columns are 'Close' and 'Volume'
        period (int): The period for calculating the sum of up and down volume (default is 50)

    Returns:
        pandas.DataFrame: A new DataFrame with an additional column 'U/D Ratio' containing the calculated Up/Down Volume Ratio values
    """
    close = df['Close']
    volume = df['Volume']
    
    # Calculate up days and down days
    upday = close > close.shift(1)
    downday = ~upday
    
    # Calculate up volume and down volume
    upvol = (upday * volume).rolling(window=period).sum()
    downvol = (downday * volume).rolling(window=period).sum()
    
    # Calculate U/D ratio (SafeDivide equivalent)
    udratio = np.where(downvol != 0, upvol / downvol, np.inf)
    
    # Add U/D Ratio to the dataframe
    df['U/D Ratio'] = udratio
    
    return df

def add_base_count(df):
    """
    Add a BaseCount column to the DataFrame, counting the streak of subsequently higher consolidations.
    The base count resets when a new consolidation undercuts the low of the previous base.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the stock's historical data with consolidation information.
            Required columns: 'Consol_Detected', 'Consol_LHS_Price', 'Consol_Depth_Percent', 'Low'

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'BaseCount' column.
    """
    # Check if required columns exist
    required_columns = ['Consol_Detected', 'Consol_LHS_Price', 'Consol_Depth_Percent', 'Low']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

    # Initialize variables
    base_count = 0
    last_base_lhs = 0
    last_base_low = 0
    current_base_low = 0
    base_counts = []
    in_consolidation = False

    for _, row in df.iterrows():
        if row['Consol_Detected']:
            current_lhs = row['Consol_LHS_Price']
            current_base_low = current_lhs * (1 - row['Consol_Depth_Percent'] / 100)
            
            if not in_consolidation:
                # Start of a new consolidation
                if base_count == 0 or current_lhs >= last_base_lhs:
                    base_count += 1
                else:
                    base_count = 1
                
                in_consolidation = True
            else:               
                # Check if current low undercuts the last completed base's low
                if row['Low'] < last_base_low:
                    base_count = 1
                    last_base_lhs = current_lhs
                    last_base_low = current_base_low
        else:
            if in_consolidation:
                # End of a consolidation
                last_base_low = current_base_low
                in_consolidation = False

        base_counts.append(base_count)

    # Add BaseCount column to the DataFrame
    df['BaseCount'] = base_counts

    return df