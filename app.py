from flask import Flask, render_template, request, jsonify
import pandas as pd
from sqlalchemy import create_engine, text
import urllib
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# Database connection parameters
params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};"
                                 "SERVER=FELIX\\OPTIONCHAIN;"
                                 "DATABASE=NiftyData;"
                                 "Trusted_Connection=yes;")

# Enhanced connection with connection pooling
engine = create_engine(
    "mssql+pyodbc:///?odbc_connect={}".format(params),
    pool_size=10,           # Maximum number of connections in the pool
    max_overflow=20         # Maximum number of overflow connections
)

# Function to execute SQL queries safely
def execute_query(query, params=None):
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), con=conn, params=params)
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Function to create Plotly chart
def create_plot(x, y_data, labels, colors):
    fig = go.Figure()
    for y, label, color in zip(y_data, labels, colors):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=label, line=dict(color=color)))
    fig.update_layout(title='', xaxis_title='', yaxis_title='', width=750, height=400,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      autosize=True)
    return pio.to_html(fig, full_html=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    query = "SELECT DISTINCT Date, strikePrice FROM option_chain_historical"
    df = execute_query(query)

    unique_dates = df['Date'].unique()
    unique_strike_prices = sorted(df['strikePrice'].unique())

    return render_template('index.html', unique_dates=unique_dates, unique_strike_prices=unique_strike_prices)

@app.route('/update_chart', methods=['POST'])
def update_chart():
    date = request.form.get('date')
    strike_price = request.form.get('strikePrice')

    query = "SELECT * FROM option_chain_historical WHERE Date = :date"
    params = {'date': date}
    if strike_price != 'All':
        query += " AND strikePrice = :strikePrice"
        params['strikePrice'] = strike_price

    df = execute_query(query, params)

    if df.empty:
        return jsonify({'plot_html': '', 'message': 'No data available for the selected filters.'})

    df = df.rename(columns={'CE.changeinOpenInterest': 'CE', 'PE.changeinOpenInterest': 'PE'})
    df_grouped = df.groupby('Time').agg({'CE': 'sum', 'PE': 'sum'}).reset_index()

    plot_html = create_plot(df_grouped['Time'], [df_grouped['CE'], df_grouped['PE']],
                            ['CE', 'PE'], ['green', 'red'])
    return jsonify({'plot_html': plot_html, 'message': ''})

@app.route('/update_ratio_chart', methods=['POST'])
def update_ratio_chart():
    date = request.form.get('date')
    query = "SELECT * FROM option_chain_historical WHERE Date = :date"
    df = execute_query(query, {'date': date})

    if df.empty:
        return jsonify({'plot_html': '', 'message': 'No data available for the selected filters.'})

    df_grouped = df.groupby('Time').agg({
        'CE.changeinOpenInterest': 'sum',
        'PE.changeinOpenInterest': 'sum',
        'CE.totalTradedVolume': 'sum',
        'PE.totalTradedVolume': 'sum'
    }).reset_index()

    df_grouped = df_grouped.replace(0, float('nan'))
    df_grouped['ChangeInOpenInterestRatio'] = df_grouped['PE.changeinOpenInterest'] / df_grouped['CE.changeinOpenInterest']
    df_grouped['TotalTradedVolumeRatio'] = df_grouped['CE.totalTradedVolume'] / df_grouped['PE.totalTradedVolume']

    plot_html = create_plot(df_grouped['Time'],
                            [df_grouped['ChangeInOpenInterestRatio'], df_grouped['TotalTradedVolumeRatio']],
                            ['PCR-COI', 'CPR-V'], ['blue', 'black'])
    return jsonify({'plot_html': plot_html, 'message': ''})

@app.route('/update_third_chart', methods=['POST'])
def update_third_chart():
    date = request.form.get('date')
    query = "SELECT * FROM option_chain_historical WHERE Date = :date"
    df = execute_query(query, {'date': date})

    if df.empty:
        return jsonify({'plot_html': '', 'message': 'No data available for the selected filters.'})

    df_grouped = df.groupby('Time').agg({
        'CE.totalTradedVolume': ['max', lambda x: x.nlargest(2).iloc[-1]],
        'CE.changeinOpenInterest': ['max', lambda x: x.nlargest(2).iloc[-1]]
    }).reset_index()

    df_grouped.columns = ['Time', 'MaxVolumeCE', 'SecondMaxVolumeCE', 'MaxOICE', 'SecondMaxOICE']
    df_grouped = df_grouped.replace(0, float('nan'))
    df_grouped['VolumeCEPercentage'] = (df_grouped['SecondMaxVolumeCE'] / df_grouped['MaxVolumeCE']) * 100
    df_grouped['OICEPercentage'] = (df_grouped['SecondMaxOICE'] / df_grouped['MaxOICE']) * 100

    plot_html = create_plot(df_grouped['Time'],
                            [df_grouped['VolumeCEPercentage'], df_grouped['OICEPercentage']],
                            ['Volume CE %', 'OI CE %'], ['black', 'green'])
    return jsonify({'plot_html': plot_html, 'message': ''})

@app.route('/update_fourth_chart', methods=['POST'])
def update_fourth_chart():
    date = request.form.get('date')
    query = "SELECT * FROM option_chain_historical WHERE Date = :date"
    df = execute_query(query, {'date': date})

    if df.empty:
        return jsonify({'plot_html': '', 'message': 'No data available for the selected filters.'})

    df_grouped = df.groupby('Time').agg({
        'PE.totalTradedVolume': ['max', lambda x: x.nlargest(2).iloc[-1]],
        'PE.changeinOpenInterest': ['max', lambda x: x.nlargest(2).iloc[-1]]
    }).reset_index()

    df_grouped.columns = ['Time', 'MaxVolumePE', 'SecondMaxVolumePE', 'MaxOIPE', 'SecondMaxOIPE']
    df_grouped = df_grouped.replace(0, float('nan'))
    df_grouped['VolumePEPercentage'] = (df_grouped['SecondMaxVolumePE'] / df_grouped['MaxVolumePE']) * 100
    df_grouped['OIPEPercentage'] = (df_grouped['SecondMaxOIPE'] / df_grouped['MaxOIPE']) * 100

    plot_html = create_plot(df_grouped['Time'],
                            [df_grouped['VolumePEPercentage'], df_grouped['OIPEPercentage']],
                            ['Volume PE %', 'OI PE %'], ['black', 'red'])
    return jsonify({'plot_html': plot_html, 'message': ''})

if __name__ == '__main__':
    app.run(debug=False)
