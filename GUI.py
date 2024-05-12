from    bokeh.io        import show
from    bokeh.plotting  import figure
from    bokeh.models    import ColumnDataSource, LinearAxis, SingleIntervalTicker, CustomJSTickFormatter, Arrow, VeeHead, BoxAnnotation, Span, Label

import  pandas  as  pd

def Launch_GUI(df):
    """
    This function creates a candlestick plot with with arrows showing:
    - Standard candlestick chart 
    - Peaks and swing high/lows denoted by arrows
    - Consolidation regions
    
    Parameters:
    df (pandas.DataFrame): The input dataframe. It should contain the following columns: 'DateString', 'Open', 'High', 'Low', 'Close', 'SwHL', 'Peak', 'Consol_Detected', 'Consol_LHS_Price', 'Consol_Depth_Percent', 'Symbol'.
    
    Returns:
    None. The function will display the plot but does not return anything.
    
    Raises:
    ValueError: If the input dataframe does not contain the required columns.
    """

    arrow_distance_pct  = 1
    arrow_length_pct    = 4
    
    # Check if the dataframe has the required columns
    required_columns = [
        'DateString', 'Open', 'High', 'Low', 'Close', 'SwHL', 'Peak', 'Consol_Detected', 'Consol_LHS_Price',
        'Consol_Depth_Percent', 'Symbol', 'Close_21_bar_ema', 'Close_50_bar_sma', 'Close_150_bar_sma', 'Close_200_bar_sma'
    ]
    if not all(column in df.columns for column in required_columns):
        raise ValueError("The input dataframe does not contain all the required columns.")
    
    # Create a dictionary to store labels for Fridays and Thursdays that are followed by a Monday
    friday_labels = {}
    for i, date in enumerate(df['DateString']):
        date = pd.to_datetime(date)
        if date.weekday() == 4:  # Friday
            friday_labels[i] = date.strftime('%Y-%m-%d')
        elif date.weekday() == 3:  # Thursday
            # Check if the next day is Monday (skipped)
            if i < len(df['DateString']) - 1 and pd.to_datetime(df['DateString'][i + 1]).weekday() == 0:
                friday_labels[i] = date.strftime('%Y-%m-%d')

    # Create a ColumnDataSource for candlesticks
    source = ColumnDataSource(data=dict(
        index=df.index.tolist(),
        Open=df['Open'],
        High=df['High'],
        Low=df['Low'],
        Close=df['Close'],
        color=['red' if close < open else 'blue' for open, close in zip(df['Open'], df['Close'])]
    ))

    # Create specific data sources for arrows with adjustments - SwHL
    up_source_SwHL = ColumnDataSource(data={
    'index':     df[(df['SwHL'] == 1) & (df['Peak'] == 0)].index,
    'High':      df[(df['SwHL'] == 1) & (df['Peak'] == 0)]['High'],
    'ArrowTip':  df[(df['SwHL'] == 1) & (df['Peak'] == 0)]['High'] * (1 + arrow_distance_pct/100),
    'ArrowTail': df[(df['SwHL'] == 1) & (df['Peak'] == 0)]['High'] * (1 + (arrow_distance_pct + arrow_length_pct)/100)
    })

    down_source_SwHL = ColumnDataSource(data={
        'index':     df[(df['SwHL'] == -1) & (df['Peak'] == 0)].index,
        'Low':       df[(df['SwHL'] == -1) & (df['Peak'] == 0)]['Low'],
        'ArrowTip':  df[(df['SwHL'] == -1) & (df['Peak'] == 0)]['Low'] * (1 - arrow_distance_pct/100),
        'ArrowTail': df[(df['SwHL'] == -1) & (df['Peak'] == 0)]['Low'] * (1 - (arrow_distance_pct + arrow_length_pct)/100)
    })

    # Create specific data sources for arrows with adjustments - peaks
    up_source_Peaks = ColumnDataSource(data={
        'index':     df[df['Peak'] == 1].index,
        'High':      df[df['Peak'] == 1]['High'],
        'ArrowTip':  df[df['Peak'] == 1]['High'] * (1 + arrow_distance_pct/100),
        'ArrowTail': df[df['Peak'] == 1]['High'] * (1 + (arrow_distance_pct + arrow_length_pct)/100)
    })

    down_source_Peaks = ColumnDataSource(data={
        'index':     df[df['Peak'] == -1].index,
        'Low':       df[df['Peak'] == -1]['Low'],
        'ArrowTip':  df[df['Peak'] == -1]['Low'] * (1 - arrow_distance_pct/100),
        'ArrowTail': df[df['Peak'] == -1]['Low'] * (1 - (arrow_distance_pct + arrow_length_pct)/100)
    })

    # Bokeh plot setup
    p = figure(x_axis_type=None, title=df['Symbol'][0], sizing_mode="stretch_width", height=900)
    p.title.text_font_size = '48pt'
    p.title.text_color = 'blue'
    p.background_fill_color = "#F2E7D4"
    p.background_fill_alpha = 0.3
    p.segment('index', 'High', 'index', 'Low', color="black", source=source)
    p.vbar('index', width=0.7, top='Open', bottom='Close', fill_color='color', line_color='color', source=source)

    # Add arrows for swing high low
    p.add_layout(Arrow(end=VeeHead(size=10), line_color="black", x_start='index', y_start='ArrowTail', x_end='index', y_end='ArrowTip', source=up_source_SwHL))
    p.add_layout(Arrow(end=VeeHead(size=10), line_color="black", x_start='index', y_start='ArrowTail', x_end='index', y_end='ArrowTip', source=down_source_SwHL))

    # Add arrows for up and down peaks
    p.add_layout(Arrow(end=VeeHead(size=14), line_color="orange",
                       x_start='index', y_start='ArrowTail', x_end='index', y_end='ArrowTip', source=up_source_Peaks))
    p.add_layout(Arrow(end=VeeHead(size=14), line_color="orange",
                       x_start='index', y_start='ArrowTail', x_end='index', y_end='ArrowTip', source=down_source_Peaks))
    
    # Add labels for high prices at peaks
    for idx, high, adj_high in zip(up_source_Peaks.data['index'], up_source_Peaks.data['High'], up_source_Peaks.data['ArrowTail']):
        formatted_high = "${:0.2f}".format(high)  # Format the price
        price_label = Label(x=idx, y=adj_high, text=formatted_high, text_font_size="8pt", text_color="green",
                            text_baseline="bottom", text_align="center")
        p.add_layout(price_label)

    # Add labels for low prices at troughs
    for idx, low, adj_low in zip(down_source_Peaks.data['index'], down_source_Peaks.data['Low'], down_source_Peaks.data['ArrowTail']):
        formatted_low = "${:0.2f}".format(low)  # Format the price
        price_label = Label(x=idx, y=adj_low, text=formatted_low, text_font_size="8pt", text_color="red",
                            text_baseline="top", text_align="center")
        p.add_layout(price_label)

    # Draw rectangles for consolidation detection
    change = df['Consol_Detected'].astype(int).diff().fillna(0) != 0
    start_indices = df.index[change & (df['Consol_Detected'])].tolist()
    # Check if the last value of 'Consol_Detected' is True, if so append the last index to end_indices
    end_indices = df.index[change & (~df['Consol_Detected'])].tolist() + [df.index[-1]]  # Append the last index if the last segment is True

    # Ensure each start has an end
    start_indices = start_indices[:len(end_indices)]

    for end in end_indices:
        right_side = end - 1  # Find the last bar where Consol_Detected is True
        left_side = right_side - df.loc[right_side, 'Consol_Len_Bars'] - 1
        top_side = df.loc[right_side, 'Consol_LHS_Price']
        bottom_side = top_side - (top_side * df.loc[right_side, 'Consol_Depth_Percent'] / 100)
        box = BoxAnnotation(left=left_side, right=right_side, top=top_side, bottom=bottom_side, fill_alpha=0.4, fill_color='orange')

        print(f"Left Side Index: {left_side}")
        print(f"Right Side Index: {right_side}")
        print(f"Consol_LHS_Price at {right_side}: {df.loc[right_side, 'Consol_LHS_Price']}")
        print(f"Consol_Depth_Percent at {right_side}: {df.loc[right_side, 'Consol_Depth_Percent']}")
        print(f"Top Side: {top_side}, Bottom Side: {bottom_side}")
        print(f"End Index: {end}\n")


        p.add_layout(box)

    # Plot SMAs and EMAs
    if 'Close_21_bar_ema' in df.columns and df['Close_21_bar_ema'].notna().any():
        p.line(df.index, df['Close_21_bar_ema'], line_width=2, color='green', legend_label='21-day EMA')
    if 'Close_50_bar_sma' in df.columns and df['Close_50_bar_sma'].notna().any():
        p.line(df.index, df['Close_50_bar_sma'], line_width=2, color='red', legend_label='50-day SMA')
    if 'Close_150_bar_sma' in df.columns and df['Close_150_bar_sma'].notna().any():
        p.line(df.index, df['Close_150_bar_sma'], line_width=2, color='blue', legend_label='150-day SMA')
    if 'Close_200_bar_sma' in df.columns and df['Close_200_bar_sma'].notna().any():
        p.line(df.index, df['Close_200_bar_sma'], line_width=3, color='black', legend_label='200-day SMA')

    # Custom x-axis configuration
    p.add_layout(LinearAxis(), 'below')
    p.xaxis.ticker = SingleIntervalTicker(interval=1)
    p.xaxis.formatter = CustomJSTickFormatter(code=f"""
        var labels = {friday_labels};
        return labels[tick] || '';
    """)

    # Assuming 'friday_labels' has indices of all Fridays
    for index in friday_labels.keys():
        friday_line = Span(location=index,  # the index of the Friday
                           dimension='height',  # Line will span the full height of the plot
                           line_color='grey',  # Color of the line
                           line_dash='solid',  # Style of the line
                           line_width=0.2)  # Width of the line
        p.add_layout(friday_line)

    p.xaxis.major_tick_line_color = 'Black'
    p.xaxis.minor_tick_line_color = None
    p.xaxis.major_label_orientation = 3.14/4  # 'vertical' or 3.14/4 for diagonal

    # legend params
    p.legend.location = 'top_left'
    p.legend.title = 'Legend'
    p.legend.title_text_font_style = "bold"
    p.legend.title_text_font_size = "12pt"
    p.legend.label_text_font_size = "10pt"


    # Display the plot
    show(p)