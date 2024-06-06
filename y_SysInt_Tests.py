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
update_RS   = True

# clear terminal
os.system('cls')

# set display options for pandas
pd.set_option('display.max_rows',       500)
pd.set_option('display.max_columns',    100)
pd.set_option('display.width',          2000)

if __name__ == '__main__':
    # get the data from Norgate
    df = fetch_OHLCV(symbol = 'DELL', num_bars = 252*2, interval = 'D')

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

        with open('us_equities_data.csv', 'w', newline='') as f:
            print("Saving us_equities_data.csv file")
            fieldnames = ['symbol', 'Security Name', 'domicile', 'Short Exchange Name', 'GICS Sector', 'GICS Industry']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()  # Write the header row
            for row in us_equities_data:
                writer.writerow(row)  # Write each dictionary as a separate row
                sec_list.append(row['symbol'])
    else:
        # Open the CSV file
        sec_list = []
        print("Loading us_equities_data.csv file")
        with open('us_equities_data.csv', 'r') as f:
            
            reader = csv.reader(f)

            # Skip the header row
            next(reader)

            # Store the data in a list of lists
            us_equities_data = [row for row in reader]

            for row in us_equities_data:
                sec_list.append(row[0])

    print(f"Total filtered securities: {len(sec_list)}")

    if update_RS:
        RS_LR = Compute_Rel_Strength_LR(sec_list, num_bars=69)
        
        # Save the DataFrame to a CSV file using to_csv method
        print("Saving RS.csv file")
        RS_LR.to_csv('RS.csv', index=True, header=True)
    else:
        # Load the DataFrame from a CSV file using pandas
        print("Loading RS.csv file")
        RS_LR = pd.read_csv('RS.csv', index_col=0)

    #df = update_stock_dataframe_with_rs(df, RS_LR, window=63)
    #df = get_stage2_uptrend(df)

    # Launch GUI
    #print("Launching GUI")
    Launch_GUI(df)

    # save df for inspection
    print("Saving df.csv")
    df.to_csv('df.csv')