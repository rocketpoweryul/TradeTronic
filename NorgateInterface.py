import norgatedata
import logging
from tqdm import tqdm
import concurrent.futures

def fetch_OHLCV(symbol          = 'GOOG', 
                num_bars         = 250, 
                interval        = 'D', 
                price_adjust    = norgatedata.StockPriceAdjustmentType.TOTALRETURN, 
                padding         = norgatedata.PaddingType.NONE):
    """
    Fetches OHLCV data for a given stock symbol from Norgate Data.
    
    Parameters:
        symbol (str): Stock symbol to fetch data for.
        num_bars (int): Number of bars to fetch.
        interval (str): Interval for the data ('D' for daily, etc.).
        price_adjust (enum): Price adjustment setting.
        padding (enum): Padding setting for missing data.

    Returns:
        pandas.DataFrame: DataFrame containing the OHLCV data with an additional 'Symbol' column.
    
    Raises:
        Exception: If the data fetching encounters an error.
    """
    try:
        OHLCV_df = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting  = price_adjust,
            padding_setting                 = padding,
            timeseriesformat                = 'pandas-dataframe',
            limit                           = num_bars,
            interval                        = interval
        )
        if OHLCV_df.empty:
            logging.warning(f"No data returned for {symbol}")
        else:
            # Add 'Symbol' column with the stock symbol
            OHLCV_df['Symbol'] = symbol
        return OHLCV_df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        raise

def get_us_equities_data():
    """
    Retrieves a list of symbols in the 'US Equities' database with detailed information for each symbol.

    For each symbol, the following details are retrieved:
    - symbol
    - Security Name
    - domicile
    - Short Exchange name
    - GICS Sector 
    - GICS Industry

    Returns:
        list: A list of dictionaries, each containing details about a US equity.

    Raises:
        Exception: If there is an issue retrieving the list of symbols or detailed information for any symbol.
    """
    def get_symbol_details(symbol):
        try:
            security_name = norgatedata.security_name(symbol)
            domicile = norgatedata.domicile(symbol)
            exchange_name = norgatedata.exchange_name(symbol)
            gics_sector = norgatedata.classification_at_level(symbol, 'GICS', 'Name', 1)
            gics_industry = norgatedata.classification_at_level(symbol, 'GICS', 'Name', 4)

            return {
                'symbol': symbol,
                'Security Name': security_name,
                'domicile': domicile,
                'Short Exchange Name': exchange_name,
                'GICS Sector': gics_sector,
                'GICS Industry': gics_industry
            }
        except Exception as e:
            print(f"Error retrieving data for symbol {symbol}: {e}")
            return None

    try:
        # Attempt to retrieve all symbols in the "US Equities" database
        symbols = norgatedata.database_symbols('US Equities')
    except Exception as e:
        # Print an error message if symbol retrieval fails and return an empty list
        print(f"Error retrieving symbols: {e}")
        return []

    # Initialize an empty list to store equity data
    equities_data = []

    # Get the total number of symbols
    total_symbols = len(symbols)

    # Use a higher number of workers if the API and system can handle it
    max_workers = 20

    excluded_keywords = ['warrant', 'cumul', 'perp', 'redeem', 'due']

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(get_symbol_details, symbol): symbol for symbol in symbols}
        for future in tqdm(concurrent.futures.as_completed(future_to_symbol), total=total_symbols, desc="Processing symbols"):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data and not any(excluded_keyword.lower() in data['Security Name'].lower() for excluded_keyword in excluded_keywords) and not data['symbol'].endswith('.U'):
                    equities_data.append(data)
            except Exception as e:
                print(f"Error processing symbol {symbol}: {e}")

    # Return the list of dictionaries containing equity information
    return equities_data
