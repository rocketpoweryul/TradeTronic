import norgatedata
import pandas as pd
import logging

def fetch_OHLCV(symbol          = 'GOOG', 
                numbars         = 250, 
                interval        = 'D', 
                price_adjust    = norgatedata.StockPriceAdjustmentType.TOTALRETURN, 
                padding         = norgatedata.PaddingType.NONE):
    """
    Fetches OHLCV data for a given stock symbol from Norgate Data.
    
    Parameters:
        symbol (str): Stock symbol to fetch data for.
        numbars (int): Number of bars to fetch.
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
            limit                           = numbars,
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
