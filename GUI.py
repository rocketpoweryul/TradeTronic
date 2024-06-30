from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearAxis, SingleIntervalTicker, CustomJSTickFormatter
from bokeh.models import Arrow, VeeHead, BoxAnnotation, Span, Label, Range1d, TextInput
from bokeh.layouts import column
import pandas as pd

def Launch_GUI(df):
    # Configuration
    arrow_distance_pct = 1
    arrow_length_pct = 4

    # Create labels for Fridays and Thursdays followed by Mondays
    friday_labels = {i: pd.to_datetime(date).strftime('%Y-%m-%d') for i, date in enumerate(df['DateString']) 
                     if pd.to_datetime(date).weekday() == 4 or 
                     (pd.to_datetime(date).weekday() == 3 and i < len(df['DateString']) - 1 
                      and pd.to_datetime(df['DateString'][i + 1]).weekday() == 0)}

    # Create ColumnDataSource for candlesticks
    source = ColumnDataSource(data=dict(
        index=df.index.tolist(),
        Open=df['Open'],
        High=df['High'],
        Low=df['Low'],
        Close=df['Close'],
        color=['red' if close < open else 'blue' for open, close in zip(df['Open'], df['Close'])]
    ))

    # Create sources for arrows
    up_source_SwHL = ColumnDataSource(data={
        'index': df[(df['SwHL'] == 1) & (df['Peak'] == 0)].index,
        'High': df[(df['SwHL'] == 1) & (df['Peak'] == 0)]['High'],
        'ArrowTip': df[(df['SwHL'] == 1) & (df['Peak'] == 0)]['High'] * (1 + arrow_distance_pct/100),
        'ArrowTail': df[(df['SwHL'] == 1) & (df['Peak'] == 0)]['High'] * (1 + (arrow_distance_pct + arrow_length_pct)/100)
    })

    down_source_SwHL = ColumnDataSource(data={
        'index': df[(df['SwHL'] == -1) & (df['Peak'] == 0)].index,
        'Low': df[(df['SwHL'] == -1) & (df['Peak'] == 0)]['Low'],
        'ArrowTip': df[(df['SwHL'] == -1) & (df['Peak'] == 0)]['Low'] * (1 - arrow_distance_pct/100),
        'ArrowTail': df[(df['SwHL'] == -1) & (df['Peak'] == 0)]['Low'] * (1 - (arrow_distance_pct + arrow_length_pct)/100)
    })

    up_source_Peaks = ColumnDataSource(data={
        'index': df[df['Peak'] == 1].index,
        'High': df[df['Peak'] == 1]['High'],
        'ArrowTip': df[df['Peak'] == 1]['High'] * (1 + arrow_distance_pct/100),
        'ArrowTail': df[df['Peak'] == 1]['High'] * (1 + (arrow_distance_pct + arrow_length_pct)/100)
    })

    down_source_Peaks = ColumnDataSource(data={
        'index': df[df['Peak'] == -1].index,
        'Low': df[df['Peak'] == -1]['Low'],
        'ArrowTip': df[df['Peak'] == -1]['Low'] * (1 - arrow_distance_pct/100),
        'ArrowTail': df[df['Peak'] == -1]['Low'] * (1 - (arrow_distance_pct + arrow_length_pct)/100)
    })

    # Main chart setup
    p = figure(x_axis_type=None, title=df['Symbol'][0], sizing_mode="stretch_width", height=500)
    p.title.text_font_size = '24pt'
    p.title.text_color = 'blue'
    p.background_fill_color = "#F2E7D4"
    p.background_fill_alpha = 0.3

    # Add candlesticks
    p.segment('index', 'High', 'index', 'Low', color="black", source=source)
    p.vbar('index', width=0.7, top='Open', bottom='Close', fill_color='color', line_color='color', source=source)

    # Add arrows
    p.add_layout(Arrow(end=VeeHead(size=10), line_color="black", x_start='index', y_start='ArrowTail', x_end='index', y_end='ArrowTip', source=up_source_SwHL))
    p.add_layout(Arrow(end=VeeHead(size=10), line_color="black", x_start='index', y_start='ArrowTail', x_end='index', y_end='ArrowTip', source=down_source_SwHL))
    p.add_layout(Arrow(end=VeeHead(size=14), line_color="orange", x_start='index', y_start='ArrowTail', x_end='index', y_end='ArrowTip', source=up_source_Peaks))
    p.add_layout(Arrow(end=VeeHead(size=14), line_color="orange", x_start='index', y_start='ArrowTail', x_end='index', y_end='ArrowTip', source=down_source_Peaks))

    # Add price labels
    for idx, high, adj_high in zip(up_source_Peaks.data['index'], up_source_Peaks.data['High'], up_source_Peaks.data['ArrowTail']):
        p.add_layout(Label(x=idx, y=adj_high, text="${:0.2f}".format(high), text_font_size="8pt", text_color="green", text_baseline="bottom", text_align="center"))
    for idx, low, adj_low in zip(down_source_Peaks.data['index'], down_source_Peaks.data['Low'], down_source_Peaks.data['ArrowTail']):
        p.add_layout(Label(x=idx, y=adj_low, text="${:0.2f}".format(low), text_font_size="8pt", text_color="red", text_baseline="top", text_align="center"))

    # Add consolidation boxes
    change = df['Consol_Detected'].astype(int).diff().fillna(0) != 0
    start_indices = df.index[change & (df['Consol_Detected'])].tolist()
    end_indices = df.index[change & (~df['Consol_Detected'])].tolist() + [df.index[-1]]
    start_indices = start_indices[:len(end_indices)]

    for end in end_indices:
        right_side = end if end == df.index[-1] else end - 1
        left_side = right_side - df.loc[right_side, 'Consol_Len_Bars'] - 1
        top_side = df.loc[right_side, 'Consol_LHS_Price']
        bottom_side = top_side - (top_side * df.loc[right_side, 'Consol_Depth_Percent'] / 100)
        p.add_layout(BoxAnnotation(left=left_side, right=right_side, top=top_side, bottom=bottom_side, fill_alpha=0.4, fill_color='orange'))

    # Add moving averages
    for ma, color in [('Close_21_bar_ema', 'green'), ('Close_50_bar_sma', 'red'), ('Close_150_bar_sma', 'blue'), ('Close_200_bar_sma', 'black')]:
        p.line(df.index, df[ma], line_width=2, color=color, legend_label=ma.replace('Close_', '').replace('_', ' '))

    # Add RSL
    last_low = df['Low'].iloc[-1]
    last_rsl = df['RSL'].iloc[-1]
    desired_rsl_position = last_low * 0.8
    rsl_scale_factor = desired_rsl_position / last_rsl
    scaled_rsl = df['RSL'] * rsl_scale_factor
    p.line(df.index, scaled_rsl, line_width=1, color='blue', legend_label='RSL')

    # Add RSL New Highs
    nh_indices = df.index[df['RSL_NH']].tolist()
    nh_values = scaled_rsl[df['RSL_NH']]
    colors = ['cyan' if consol else 'lightblue' for consol in df.loc[df['RSL_NH'], 'Consol_Detected']]
    sizes = [10 if consol else 7 for consol in df.loc[df['RSL_NH'], 'Consol_Detected']]
    p.scatter(nh_indices, nh_values, size=sizes, color=colors, alpha=0.8)

    # Legend setup
    p.legend.location = 'top_left'
    p.legend.title = 'Legend'
    p.legend.title_text_font_style = "bold"
    p.legend.title_text_font_size = "12pt"
    p.legend.label_text_font_size = "10pt"

    # Subplots setup
    subplot_height = 150
    subplot_params = [
        ('Stage 2', 'green', 0, 1),
        ('Consol_Detected', 'orange', 0, 1),
        ('UpDownVolumeRatio', 'blue', None, None),
        ('ATR', 'red', None, None),
        ('%B', 'purple', 0, 1),
        ('Williams %R', 'brown', -100, 0),  # New subplot for Williams %R
        ('ADR', 'teal', None, None),  # New subplot for ADR
        ('U/D Ratio', 'magenta', None, None),  # New subplot for Up/Down Ratio
        ('BaseCount', 'gold', 0, None)  # New subplot for BaseCount
    ]

    subplots = []

    for param, color, y_min, y_max in subplot_params:
        sub_p = figure(x_range=p.x_range, height=subplot_height, title=param, sizing_mode="stretch_width", x_axis_type=None)
        
        if param in ['Stage 2', 'Consol_Detected', 'BaseCount']:
            sub_p.step(df.index, df[param], line_color=color, mode="after")
        else:
            sub_p.line(df.index, df[param], line_color=color)
        
        if y_min is not None and y_max is not None:
            sub_p.y_range = Range1d(y_min, y_max)
        
        sub_p.xaxis.major_label_orientation = 3.14/4
        
        for index in friday_labels.keys():
            sub_p.add_layout(Span(location=index, dimension='height', line_color='grey', line_dash='solid', line_width=0.2))
        
        subplots.append(sub_p)

    # Add x-axis to the last subplot
    subplots[-1].xaxis.ticker = SingleIntervalTicker(interval=1)
    subplots[-1].xaxis.formatter = CustomJSTickFormatter(code=f"""
        var labels = {friday_labels};
        return labels[tick] || '';
    """)

    # Create the input field
    ticker_input = TextInput(value="", title="Ticker Symbol:")

    # Combine main plot and subplots
    layout = column(p, *subplots, sizing_mode="stretch_width")

    # Display the plot
    show(layout)