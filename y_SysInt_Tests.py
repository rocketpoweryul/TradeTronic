# open source modules
import pandas as pd
import os
import csv

# functions from this project
from NorgateInterface import *
from TATools          import *
from GUI              import *

# perform daily maintenance on data stored to disk
update_useq = False
update_RS   = False

# clear terminal
os.system('cls')

# set display options for pandas
pd.set_option('display.max_rows',       500)
pd.set_option('display.max_columns',    100)
pd.set_option('display.width',          2000)

# get the data from Norgate
df = fetch_OHLCV(symbol = 'DELL', num_bars = 252, interval = 'D')

## Initial stock dataframe processeing
df = find_swing_high_and_lows(df)
df = filter_swing_high_and_lows(df)
df = filter_peaks(df)
df = detect_consolidation(df)
df = add_moving_average(df,  21, 'ema')
df = add_moving_average(df,  50, 'sma')
df = add_moving_average(df, 150, 'sma')
df = add_moving_average(df, 200, 'sma')
df = add_relative_strength_line(df)

# Convert index to integers, but save dates as datestrings 
df['DateString'] = df.index.strftime('%Y-%m-%d')  # Save date info before resetting index

# Determine Friday dates before resetting the index for plotting purposes
df['DayOfWeek'] = df.index.weekday  # Monday=0, Sunday=6

# Reset index to use numerical index
df.reset_index(drop=True, inplace=True)

if update_useq:
    us_equities_data = get_us_equities_data()
    sec_list = []

    # Print a chunk of us_equities_data for verification
    chunk_size = min(5, len(us_equities_data))  # Adjust chunk_size as needed
    print("Sample of us_equities_data:")
    for row in us_equities_data[:chunk_size]:
        print(row)

    with open('us_equities_data.csv', 'w', newline='') as f:
        fieldnames = ['symbol', 'Security Name', 'domicile', 'Short Exchange Name', 'GICS Sector', 'GICS Industry']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row
        for row in us_equities_data:
            writer.writerow(row)  # Write each dictionary as a separate row
            sec_list.append(row['symbol'])
else:
    # Open the CSV file
    sec_list =  []
    with open('us_equities_data.csv', 'r') as f:
        
        reader = csv.reader(f)

        # Store the data in a list of lists
        us_equities_data = [row for row in reader]

        for row in us_equities_data:
            sec_list.append(row[0])

if update_RS:
    RS_LR = Compute_Rel_Strength_LR(sec_list, num_bars=69)
    
    # Save the DataFrame to a CSV file using to_csv method
    RS_LR.to_csv('RS_LR.csv', index=True, header=True)
else:
    # Load the DataFrame from a CSV file using pandas
    RS_LR = pd.read_csv('RS_LR.csv', index_col=0)

print(df.head())

df = update_stock_dataframe_with_rs(df, RS_LR, window=63)

# Launch GUI
Launch_GUI(df)