# open source modules
from    NorgateInterface import *
import  pandas as pd
import  os

# functions from this project
from    TATools          import *
from    GUI              import *

# clear terminal
os.system('cls')

# set display options for pandas
pd.set_option('display.max_rows',       500)
pd.set_option('display.max_columns',    100)
pd.set_option('display.width',          2000)

# get the data from Norgate
df = fetch_OHLCV(symbol          = 'MMYT', 
                 numbars         = 500, 
                 interval        = 'D', 
                 price_adjust    = norgatedata.StockPriceAdjustmentType.TOTALRETURN, 
                 padding         = norgatedata.PaddingType.NONE)

## Initial stock dataframe processeing
df = find_swing_high_and_lows(df)
df = filter_swing_high_and_lows(df)
df = filter_peaks(df)
df = detect_consolidation(df)
df = add_moving_average(df,  21, 'ema')
df = add_moving_average(df,  50, 'sma')
df = add_moving_average(df, 150, 'sma')
df = add_moving_average(df, 200, 'sma')

# Convert index to integers, but save dates as datestrings 
df['DateString'] = df.index.strftime('%Y-%m-%d')  # Save date info before resetting index

# Determine Friday dates before resetting the index for plotting purposes
df['DayOfWeek'] = df.index.weekday  # Monday=0, Sunday=6

# Reset index to use numerical index
df.reset_index(drop=True, inplace=True)

# Launch GUI
Launch_GUI(df)

df.to_csv('df.csv', index=True)