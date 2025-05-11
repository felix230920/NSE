import os
import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi import Form
from contextlib import asynccontextmanager
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import Date
import urllib
import plotly.graph_objects as go
import plotly.io as pio
from cachetools import cached, TTLCache
import requests
import time
import http.cookiejar as cookielib
from datetime import datetime
import logging
import json
import locale

locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

machine_name = os.environ.get('COMPUTERNAME', 'DEFAULT')  # 'DEFAULT' is a fallback if COMPUTERNAME is not found
SERVER = f"{machine_name}\\OPTIONCHAIN"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    asyncio.create_task(background_task())  # Start the background task
    yield  # Application runs here
    # Add any shutdown logic here if needed

app = FastAPI(lifespan=lifespan)

# Specify the absolute path to the templates directory
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
templates = Jinja2Templates(directory=templates_dir)

# Database connection parameters
params = urllib.parse.quote_plus(f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                                 f"SERVER={SERVER};DATABASE=NiftyData;"
                                 f"Trusted_Connection=yes;")

# Enhanced connection with connection pooling
engine = create_engine(
    "mssql+pyodbc:///?odbc_connect={}".format(params),
    pool_size=10,           # Maximum number of connections in the pool
    max_overflow=20         # Maximum number of overflow connections
)

# Cache with a TTL of 10 seconds
cache = TTLCache(maxsize=100, ttl=10)

# Function to create a cache key from query and params
def make_cache_key(query, params):
    if params:
        return (query, tuple(sorted(params.items())))
    return query

# Function to execute SQL queries safely
@cached(cache, key=lambda query, params=None: make_cache_key(query, params))
def execute_query(query, params=None):
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), con=conn, params=params)
    except Exception as e:
        logging.error(f"Database error: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Function to create Plotly chart
def create_plot(x, y_data, labels, colors):
    fig = go.Figure()
    for y, label, color in zip(y_data, labels, colors):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=label, line=dict(color=color)))
    fig.update_layout(title='', xaxis_title='', yaxis_title='', width=750, height=400,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      autosize=True)
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the index page with expiry dates, dates, and strike prices."""
    # Fetch and sort expiry dates directly in SQL
    query = "SELECT DISTINCT expiryDate FROM option_chain_historical ORDER BY expiryDate DESC"
    df = execute_query(query)
    unique_expiry_dates = df['expiryDate'].tolist()

    # Get the most recent expiry date as the default
    default_expiry_date = unique_expiry_dates[0] if unique_expiry_dates else None

    # Fetch dates and strike prices for the default expiry date
    unique_dates, default_date, strike_prices = fetch_dates_and_strike_prices(default_expiry_date)

    # Render the template with the fetched data
    return templates.TemplateResponse(
        'index.html',
        {
            "request": request,
            "unique_expiry_dates": unique_expiry_dates,
            "default_expiry_date": default_expiry_date,
            "unique_dates": unique_dates,
            "default_date": default_date,
            "strike_prices": strike_prices
        }
    )


def fetch_dates_and_strike_prices(expiry_date, selected_date=None):
    """Fetch dates and strike prices for a given expiry date and optionally a selected date."""
    if not expiry_date:
        return [], None, []

    # Fetch dates for the given expiry date, sorted in SQL
    query_dates = """
        SELECT DISTINCT Date 
        FROM option_chain_historical 
        WHERE expiryDate = :expiryDate 
        ORDER BY Date DESC
    """
    df_dates = execute_query(query_dates, {'expiryDate': expiry_date})
    unique_dates = df_dates['Date'].astype(str).tolist()
    default_date = unique_dates[0] if unique_dates else None

    # Use the selected date if provided, otherwise use the default date
    date_to_use = selected_date if selected_date else default_date

    # Fetch strike prices for the selected/default date, sorted in SQL
    if date_to_use:
        query_strike_prices = """
            SELECT DISTINCT strikePrice 
            FROM option_chain_historical 
            WHERE expiryDate = :expiryDate AND Date = :date
            ORDER BY strikePrice
        """
        df_strike_prices = execute_query(query_strike_prices, {'expiryDate': expiry_date, 'date': date_to_use})
        strike_prices = df_strike_prices['strikePrice'].tolist()
    else:
        strike_prices = []

    return unique_dates, default_date, strike_prices


@app.post("/get_dates_and_strike_prices")
async def get_dates_and_strike_prices(request: Request):
    """Fetch both dates and strike prices in a single API call."""
    form_data = await request.form()
    expiry_date = form_data.get('expiryDate')
    selected_date = form_data.get('date')  # Optional selected date from the frontend

    # Fetch dates and strike prices for the given expiry date and selected date
    unique_dates, default_date, strike_prices = fetch_dates_and_strike_prices(expiry_date, selected_date)

    # Return the data as JSON
    return JSONResponse(content={
        'dates': unique_dates,
        'default_date': default_date,
        'strike_prices': strike_prices
    })

@app.post("/update_all_charts")
async def update_all_charts(request: Request):
    """Update all charts in a single API call."""
    form_data = await request.form()
    expiry_date = form_data.get('expiryDate')
    date = form_data.get('date')
    strike_price = form_data.get('strikePrice')

    # Fetch data for all charts
    chart_data = fetch_chart_data(expiry_date, date, strike_price)

    return JSONResponse(content=chart_data)

def fetch_chart_data(expiry_date, date, strike_price):
    """Fetch data for all charts in a single query."""
    # Base query for all charts
    query = """
        SELECT Time, expiryDate, strikePrice, 
               CE_changeinOpenInterest, PE_changeinOpenInterest,
               CE_totalTradedVolume, PE_totalTradedVolume,
               CE_openInterest, PE_openInterest
        FROM option_chain_historical
        WHERE expiryDate = :expiryDate AND Date = :date
    """
    params = {'expiryDate': expiry_date, 'date': date}

    # Add strikePrice condition only if it's not 'All'
    if strike_price != 'All':
        query += " AND strikePrice = :strikePrice"
        params['strikePrice'] = strike_price

    df = execute_query(query, params)

    if df.empty:
        return {
            'chart': '',
            'ratio_chart': '',
            'third_chart': '',
            'fourth_chart': ''
        }

    # Process data for all charts
    chart_data = {
        'chart': process_main_chart(df, strike_price),  # Pass strike_price for main chart
        'ratio_chart': process_ratio_chart(df),
        'third_chart': process_third_chart(df),
        'fourth_chart': process_fourth_chart(df)
    }

    return chart_data

def process_main_chart(df, strike_price):
    """Process data for the main chart."""
    # Filter data by strikePrice if it's not 'All'
    if strike_price != 'All':
        df = df[df['strikePrice'] == float(strike_price)]

    df_grouped = df.groupby('Time').agg({
        'CE_changeinOpenInterest': 'sum',
        'PE_changeinOpenInterest': 'sum'
    }).reset_index()

    return create_plot(
        df_grouped['Time'],
        [df_grouped['CE_changeinOpenInterest'], df_grouped['PE_changeinOpenInterest']],
        ['CE', 'PE'],
        ['green', 'red']
    )

def process_ratio_chart(df):
    """Process data for the ratio chart."""
    df_grouped = df.groupby('Time').agg({
        'CE_changeinOpenInterest': 'sum',
        'PE_changeinOpenInterest': 'sum',
        'CE_totalTradedVolume': 'sum',
        'PE_totalTradedVolume': 'sum'
    }).reset_index()

    # Replace zeros with NaN to avoid division errors
    df_grouped = df_grouped.replace(0, float('nan'))

    # Safely calculate ratios, handling division by zero
    df_grouped['ChangeInOpenInterestRatio'] = df_grouped['PE_changeinOpenInterest'] / df_grouped['CE_changeinOpenInterest']
    df_grouped['TotalTradedVolumeRatio'] = df_grouped['CE_totalTradedVolume'] / df_grouped['PE_totalTradedVolume']

    # Replace invalid values (e.g., NaN, inf) with a default value or drop them
    df_grouped['ChangeInOpenInterestRatio'] = df_grouped['ChangeInOpenInterestRatio'].replace([float('inf'), -float('inf')], float('nan'))

    return create_plot(
        df_grouped['Time'],
        [df_grouped['ChangeInOpenInterestRatio'], df_grouped['TotalTradedVolumeRatio']],
        ['PCR-COI', 'CPR-V'],
        ['blue', 'black']
    )

def process_third_chart(df):
    """Process data for the third chart."""
    df_grouped = df.groupby('Time').agg({
        'CE_totalTradedVolume': ['max', lambda x: x.nlargest(2).iloc[-1]],
        'CE_openInterest': ['max', lambda x: x.nlargest(2).iloc[-1]]
    }).reset_index()

    df_grouped.columns = ['Time', 'MaxVolumeCE', 'SecondMaxVolumeCE', 'MaxOICE', 'SecondMaxOICE']
    df_grouped = df_grouped.replace(0, float('nan'))
    df_grouped['VolumeCEPercentage'] = (df_grouped['SecondMaxVolumeCE'] / df_grouped['MaxVolumeCE']) * 100
    df_grouped['OICEPercentage'] = (df_grouped['SecondMaxOICE'] / df_grouped['MaxOICE']) * 100

    return create_plot(
        df_grouped['Time'],
        [df_grouped['VolumeCEPercentage'], df_grouped['OICEPercentage']],
        ['Volume CE %', 'OI CE %'],
        ['black', 'green']
    )

def process_fourth_chart(df):
    """Process data for the fourth chart."""
    df_grouped = df.groupby('Time').agg({
        'PE_totalTradedVolume': ['max', lambda x: x.nlargest(2).iloc[-1]],
        'PE_openInterest': ['max', lambda x: x.nlargest(2).iloc[-1]]
    }).reset_index()

    df_grouped.columns = ['Time', 'MaxVolumePE', 'SecondMaxVolumePE', 'MaxOIPE', 'SecondMaxOIPE']
    df_grouped = df_grouped.replace(0, float('nan'))
    df_grouped['VolumePEPercentage'] = (df_grouped['SecondMaxVolumePE'] / df_grouped['MaxVolumePE']) * 100
    df_grouped['OIPEPercentage'] = (df_grouped['SecondMaxOIPE'] / df_grouped['MaxOIPE']) * 100

    return create_plot(
        df_grouped['Time'],
        [df_grouped['VolumePEPercentage'], df_grouped['OIPEPercentage']],
        ['Volume PE %', 'OI PE %'],
        ['black', 'red']
    )
option_chain_cache = TTLCache(maxsize=100, ttl=10)

@app.get("/option_chain", response_class=HTMLResponse)
async def option_chain(request: Request):
    """Fetch and display the option_chain table."""
    if "option_chain" in option_chain_cache:
        return option_chain_cache["option_chain"]
    try:
        # Fetch data from the option_chain table
        query = "SELECT * FROM NiftyData.dbo.option_chain"
        df = execute_query(query)

        if df.empty:
            logging.warning("No data found in the option_chain table.")
            return HTMLResponse(content="<p>No data available in the option_chain table.</p>", status_code=200)
        
        # Find the strikePrice corresponding to the max totalTradedVolume
        ce_max_volume_index = df['CE_totalTradedVolume'].idxmax()  # Get the index of the max value
        ce_strike_price = df.loc[ce_max_volume_index, 'strikePrice']  # Get the strikePrice at that index
        strike_price_plus_25 = ce_strike_price + 25  # Add 25 to the strikePrice
        strike_price_plus_75 = ce_strike_price + 75  # Add 75 to the strikePrice
        pe_max_volume_index = df['PE_totalTradedVolume'].idxmax()  # Get the index of the max value
        pe_strike_price = df.loc[pe_max_volume_index, 'strikePrice']  # Get the strikePrice at that index
        strike_price_minus_25 = pe_strike_price - 25  # Subtract 25 to the strikePrice
        strike_price_minus_75 = pe_strike_price - 75  # Subtract 75 to the strikePrice

        # Helper function to calculate the result for each formula
        def calculate_result(column, strike_price_column):
            largest = df[column].nlargest(2)
            if len(largest) < 2:
                return "Not enough data"

            largest_value = largest.iloc[0]
            second_largest_value = largest.iloc[1]
            ratio = (second_largest_value / largest_value) * 100

            largest_strike_price = df.loc[df[column] == largest_value, strike_price_column].iloc[0]
            second_largest_strike_price = df.loc[df[column] == second_largest_value, strike_price_column].iloc[0]

            if ratio < 75:
                result = "STRONG"
            else:
                if largest_strike_price > second_largest_strike_price:
                    result = "WTB"
                else:
                    result = "WTT"

            return f"{result} {ratio:.2f}%"

        # Calculate the four values
        ce_total_traded_volume_result = calculate_result("CE_totalTradedVolume", "strikePrice")
        ce_open_interest_result = calculate_result("CE_openInterest", "strikePrice")
        pe_total_traded_volume_result = calculate_result("PE_totalTradedVolume", "strikePrice")
        pe_open_interest_result = calculate_result("PE_openInterest", "strikePrice")

        # Add custom headers for Date, Time, and Expiry Date
        date_value = df.iloc[0]['Date'] if 'Date' in df.columns else 'N/A'
        time_value = df.iloc[0]['Time'] if 'Time' in df.columns else 'N/A'
        expiry_date_value = df.iloc[0]['expiryDate'] if 'expiryDate' in df.columns else 'N/A'
        underlying_value = df.iloc[0]['underlyingValue'] if 'underlyingValue' in df.columns else 'N/A'

        # Remove unnecessary columns
        df = df.drop(columns=['Date', 'Time', 'expiryDate', 'LowerBound', 'UpperBound', 'underlyingValue', 'Identifier'], errors='ignore')
        # Reorder columns to move strikePrice to the 8th position
        if 'strikePrice' in df.columns:
            columns = list(df.columns)
            columns.remove('strikePrice')
            columns.insert(7, 'strikePrice')  # Insert strikePrice at the 8th position (index 7)
            df = df[columns]

        # Ensure numeric values are used for the calculations
        def highlight_max(s, color):
            # Convert string to numeric for proper comparison
            s_numeric = pd.to_numeric(s, errors='coerce')  # Coerce invalid values to NaN
            is_max = s_numeric == s_numeric.max()
            return [f"background-color: {color}" if v else "" for v in is_max]

        def highlight_second_high(s, color):
            # Convert string to numeric for proper comparison
            s_numeric = pd.to_numeric(s, errors='coerce')  # Coerce invalid values to NaN
            max_value = s_numeric.max()
            second_highest = s_numeric.nlargest(2).iloc[-1] if len(s_numeric.nlargest(2)) > 1 else None
            return [
                f"background-color: {color}" if v == second_highest and second_highest > 0.75 * max_value else ""
                for v in s_numeric
            ]

        # Apply formatting to the DataFrame
        styled_df = df.style.format({
            "CE_changeinOpenInterest": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['CE_changeinOpenInterest'].max():.2%})" if pd.notnull(x) else "",
            "CE_openInterest": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['CE_openInterest'].max():.2%})" if pd.notnull(x) else "",
            "CE_totalTradedVolume": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['CE_totalTradedVolume'].max():.2%})" if pd.notnull(x) else "",
            "PE_changeinOpenInterest": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['PE_changeinOpenInterest'].max():.2%})" if pd.notnull(x) else "",
            "PE_openInterest": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['PE_openInterest'].max():.2%})" if pd.notnull(x) else "",
            "PE_totalTradedVolume": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['PE_totalTradedVolume'].max():.2%})" if pd.notnull(x) else "",
        }).apply(highlight_max, color="limegreen", subset=["CE_changeinOpenInterest"]) \
        .apply(highlight_max, color="magenta", subset=["CE_openInterest"]) \
        .apply(highlight_max, color="dodgerblue", subset=["CE_totalTradedVolume"]) \
        .apply(highlight_max, color="limegreen", subset=["PE_changeinOpenInterest"]) \
        .apply(highlight_max, color="magenta", subset=["PE_openInterest"]) \
        .apply(highlight_max, color="dodgerblue", subset=["PE_totalTradedVolume"]) \
        .apply(highlight_second_high, color="yellow", subset=["CE_openInterest"]) \
        .apply(highlight_second_high, color="yellow", subset=["CE_totalTradedVolume"]) \
        .apply(highlight_second_high, color="yellow", subset=["PE_openInterest"]) \
        .apply(highlight_second_high, color="yellow", subset=["PE_totalTradedVolume"])\
        .map(lambda x: "background-color: orange; font-weight: bold;", subset=["strikePrice"])

        # Convert the styled DataFrame to an HTML table with centered alignment
        table_html = styled_df.to_html(index=False, classes="table table-striped", border=0, escape=False)
        table_html = table_html.replace('<table ', '<table style="text-align: center; width: 100%; white-space: nowrap;" ')

        time_value_str = time_value.isoformat()  # Converts to "HH:MM:SS" format
        date_value_str = date_value.isoformat() 
        expiry_date_value_str = expiry_date_value.isoformat()

        # Return both the table HTML and the underlying value
        response_data = {
            "html": table_html,
            "ce_strike_price": int(ce_strike_price),  # Convert to int
            "pe_strike_price": int(pe_strike_price),  # Convert to int
            "ce_total_traded_volume_result": ce_total_traded_volume_result,
            "ce_open_interest_result": ce_open_interest_result,
            "pe_total_traded_volume_result": pe_total_traded_volume_result,
            "pe_open_interest_result": pe_open_interest_result,
            "time_value_str": time_value_str,
            "date_value_str": date_value_str,
            "expiry_date_value_str": expiry_date_value_str,
            "underlying_value": int(underlying_value),
            "strike_price_plus_25": int(strike_price_plus_25),  # Convert to int
            "strike_price_plus_75": int(strike_price_plus_75),   # Convert to int
            "strike_price_minus_25": int(strike_price_minus_25),  # Convert to int
            "strike_price_minus_75": int(strike_price_minus_75)   # Convert to int
        }
        
        option_chain_cache["option_chain"] = JSONResponse(content=jsonable_encoder(response_data), status_code=200)
        return option_chain_cache["option_chain"]

    except Exception as e:
        logging.error(f"Error fetching option_chain data: {e}")
        response_data = {
            "html": "<p>An error occurred while fetching the data.</p>",
            "underlying_value": "N/A"
        }
        return JSONResponse(content=response_data, status_code=500)

@app.post("/get_historical_dates_and_times")
async def get_historical_dates_and_times(request: Request):
    """Fetch dates and times for a given expiry date and optionally a selected date."""
    form_data = await request.form()
    expiry_date = form_data.get('expiryDate')
    selected_date = form_data.get('date')  # Optional selected date from the frontend

    if not expiry_date:
        return JSONResponse(content={"dates": [], "default_date": None, "times": [], "default_time": None})

    # Fetch dates for the given expiry date
    query_dates = """
        SELECT DISTINCT Date 
        FROM option_chain_historical 
        WHERE expiryDate = :expiryDate 
        ORDER BY Date DESC
    """
    df_dates = execute_query(query_dates, {'expiryDate': expiry_date})
    unique_dates = df_dates['Date'].astype(str).tolist()
    default_date = unique_dates[0] if unique_dates else None

    # Use the selected date if provided, otherwise use the default date
    date_to_use = selected_date if selected_date else default_date

    # Fetch times for the selected/default date
    if date_to_use:
        query_times = """
            SELECT DISTINCT Time 
            FROM option_chain_historical 
            WHERE expiryDate = :expiryDate AND Date = :date 
            ORDER BY Time
        """
        df_times = execute_query(query_times, {'expiryDate': expiry_date, 'date': date_to_use})
        unique_times = df_times['Time'].astype(str).tolist()
        default_time = unique_times[0] if unique_times else None
    else:
        unique_times = []
        default_time = None

    return JSONResponse(content={
        "dates": unique_dates,
        "default_date": default_date,
        "times": unique_times,
        "default_time": default_time
    })

historical_option_chain_cache = TTLCache(maxsize=100, ttl=10)

@app.post("/historical_option_chain", response_class=HTMLResponse)
async def historical_option_chain(request: Request):
    """Fetch and display the option_chain_historical table with filters."""
    form_data = await request.form()
    expiry_date = form_data.get('expiryDate')
    date = form_data.get('date')
    time = form_data.get('time')

    try:
        # Base query with filters
        query = """
            SELECT * 
            FROM option_chain_historical 
            WHERE expiryDate = :expiryDate AND Date = :date AND Time = :time
            ORDER BY strikePrice ASC
        """
        params = {'expiryDate': expiry_date, 'date': date, 'time': time}
        df = execute_query(query, params)

        if df.empty:
            logging.warning("No data found for the selected filters.")
            return HTMLResponse(content="<p>No data available for the selected filters.</p>", status_code=200)

        # Find the strikePrice corresponding to the max totalTradedVolume
        ce_max_volume_index = df['CE_totalTradedVolume'].idxmax()  # Get the index of the max value
        ce_strike_price = df.loc[ce_max_volume_index, 'strikePrice']  # Get the strikePrice at that index
        strike_price_plus_25 = ce_strike_price + 25  # Add 25 to the strikePrice
        strike_price_plus_75 = ce_strike_price + 75  # Add 75 to the strikePrice
        pe_max_volume_index = df['PE_totalTradedVolume'].idxmax()  # Get the index of the max value
        pe_strike_price = df.loc[pe_max_volume_index, 'strikePrice']  # Get the strikePrice at that index
        strike_price_minus_25 = pe_strike_price - 25  # Subtract 25 to the strikePrice
        strike_price_minus_75 = pe_strike_price - 75  # Subtract 75 to the strikePrice

        # Helper function to calculate the result for each formula
        def calculate_result(column, strike_price_column):
            largest = df[column].nlargest(2)
            if len(largest) < 2:
                return "Not enough data"

            largest_value = largest.iloc[0]
            second_largest_value = largest.iloc[1]
            ratio = (second_largest_value / largest_value) * 100

            largest_strike_price = df.loc[df[column] == largest_value, strike_price_column].iloc[0]
            second_largest_strike_price = df.loc[df[column] == second_largest_value, strike_price_column].iloc[0]

            if ratio < 75:
                result = "STRONG"
            else:
                if largest_strike_price > second_largest_strike_price:
                    result = "WTB"
                else:
                    result = "WTT"

            return f"{result} {ratio:.2f}%"

        # Calculate the four values
        ce_total_traded_volume_result = calculate_result("CE_totalTradedVolume", "strikePrice")
        ce_open_interest_result = calculate_result("CE_openInterest", "strikePrice")
        pe_total_traded_volume_result = calculate_result("PE_totalTradedVolume", "strikePrice")
        pe_open_interest_result = calculate_result("PE_openInterest", "strikePrice")

        # Add custom headers for Date, Time, and Expiry Date
        date_value = df.iloc[0]['Date'] if 'Date' in df.columns else 'N/A'
        time_value = df.iloc[0]['Time'] if 'Time' in df.columns else 'N/A'
        expiry_date_value = df.iloc[0]['expiryDate'] if 'expiryDate' in df.columns else 'N/A'
        underlying_value = df.iloc[0]['underlyingValue'] if 'underlyingValue' in df.columns else 'N/A'

        # Remove unnecessary columns
        df = df.drop(columns=['Date', 'Time', 'expiryDate', 'LowerBound', 'UpperBound', 'underlyingValue', 'Identifier'], errors='ignore')
        # Reorder columns to move strikePrice to the 8th position
        if 'strikePrice' in df.columns:
            columns = list(df.columns)
            columns.remove('strikePrice')
            columns.insert(7, 'strikePrice')  # Insert strikePrice at the 8th position (index 7)
            df = df[columns]

        # Ensure numeric values are used for the calculations
        def highlight_max(s, color):
            # Convert string to numeric for proper comparison
            s_numeric = pd.to_numeric(s, errors='coerce')  # Coerce invalid values to NaN
            is_max = s_numeric == s_numeric.max()
            return [f"background-color: {color}" if v else "" for v in is_max]

        def highlight_second_high(s, color):
            # Convert string to numeric for proper comparison
            s_numeric = pd.to_numeric(s, errors='coerce')  # Coerce invalid values to NaN
            max_value = s_numeric.max()
            second_highest = s_numeric.nlargest(2).iloc[-1] if len(s_numeric.nlargest(2)) > 1 else None
            return [
                f"background-color: {color}" if v == second_highest and second_highest > 0.75 * max_value else ""
                for v in s_numeric
            ]

        # Apply formatting to the DataFrame
        styled_df = df.style.format({
            "CE_changeinOpenInterest": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['CE_changeinOpenInterest'].max():.2%})" if pd.notnull(x) else "",
            "CE_openInterest": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['CE_openInterest'].max():.2%})" if pd.notnull(x) else "",
            "CE_totalTradedVolume": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['CE_totalTradedVolume'].max():.2%})" if pd.notnull(x) else "",
            "PE_changeinOpenInterest": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['PE_changeinOpenInterest'].max():.2%})" if pd.notnull(x) else "",
            "PE_openInterest": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['PE_openInterest'].max():.2%})" if pd.notnull(x) else "",
            "PE_totalTradedVolume": lambda x: f"{locale.format_string('%d', x, grouping=True)} ({x / df['PE_totalTradedVolume'].max():.2%})" if pd.notnull(x) else "",
        }).apply(highlight_max, color="limegreen", subset=["CE_changeinOpenInterest"]) \
        .apply(highlight_max, color="magenta", subset=["CE_openInterest"]) \
        .apply(highlight_max, color="dodgerblue", subset=["CE_totalTradedVolume"]) \
        .apply(highlight_max, color="limegreen", subset=["PE_changeinOpenInterest"]) \
        .apply(highlight_max, color="magenta", subset=["PE_openInterest"]) \
        .apply(highlight_max, color="dodgerblue", subset=["PE_totalTradedVolume"]) \
        .apply(highlight_second_high, color="yellow", subset=["CE_openInterest"]) \
        .apply(highlight_second_high, color="yellow", subset=["CE_totalTradedVolume"]) \
        .apply(highlight_second_high, color="yellow", subset=["PE_openInterest"]) \
        .apply(highlight_second_high, color="yellow", subset=["PE_totalTradedVolume"])\
        .map(lambda x: "background-color: orange; font-weight: bold;", subset=["strikePrice"])

        # Convert the styled DataFrame to an HTML table with centered alignment
        table_html = styled_df.to_html(index=False, classes="table table-striped", border=0, escape=False)
        table_html = table_html.replace('<table ', '<table style="text-align: center; width: 100%; white-space: nowrap;" ')

        time_value_str = time_value.isoformat()  # Converts to "HH:MM:SS" format
        date_value_str = date_value.isoformat() 
        expiry_date_value_str = expiry_date_value.isoformat()

        # Return both the table HTML and the underlying value
        response_data = {
            "html": table_html,
            "ce_strike_price": int(ce_strike_price),  # Convert to int
            "pe_strike_price": int(pe_strike_price),  # Convert to int
            "ce_total_traded_volume_result": ce_total_traded_volume_result,
            "ce_open_interest_result": ce_open_interest_result,
            "pe_total_traded_volume_result": pe_total_traded_volume_result,
            "pe_open_interest_result": pe_open_interest_result,
            "time_value_str": time_value_str,
            "date_value_str": date_value_str,
            "expiry_date_value_str": expiry_date_value_str,
            "underlying_value": int(underlying_value),
            "strike_price_plus_25": int(strike_price_plus_25),  # Convert to int
            "strike_price_plus_75": int(strike_price_plus_75),   # Convert to int
            "strike_price_minus_25": int(strike_price_minus_25),  # Convert to int
            "strike_price_minus_75": int(strike_price_minus_75)   # Convert to int
        }
        
        historical_option_chain_cache["historical_option_chain"] = JSONResponse(content=jsonable_encoder(response_data), status_code=200)
        return historical_option_chain_cache["historical_option_chain"]

    except Exception as e:
        logging.error(f"Error fetching option_chain data: {e}")
        response_data = {
            "html": "<p>An error occurred while fetching the data.</p>",
            "underlying_value": "N/A"
        }
        return JSONResponse(content=response_data, status_code=500)

# LTP fetching and processing
url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
headers = {
    "Accept-Encoding": "deflate",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive"
}

# Create a session and set up cookie jar
session = requests.Session()
session.headers.update(headers)
session.cookies = cookielib.LWPCookieJar()

def update_cookies():
    session.get("https://www.nseindia.com")
    session.get("https://www.nseindia.com/option-chain")
    logging.info("Cookies updated.")

def fetch_data():
    try:
        response = session.get(url)
        response.raise_for_status()
        if 'application/json' in response.headers.get('Content-Type', ''):
            if response.content:
                decompressed_data = response.content.decode('utf-8', errors='replace')
                if decompressed_data.strip().startswith('<'):
                    logging.error("Received HTML content instead of JSON")
                    return None
                return json.loads(decompressed_data)
            else:
                logging.warning("Empty response received")
        else:
            logging.warning(f"Unexpected content type: {response.headers.get('Content-Type')}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
    return None

def process_data(data):
    try:
        option_chain_historical = data['filtered']['data']
        ce_data = [option['CE'] for option in option_chain_historical if 'CE' in option]
        pe_data = [option['PE'] for option in option_chain_historical if 'PE' in option]

        ce_df = pd.DataFrame(ce_data)
        pe_df = pd.DataFrame(pe_data)

        ce_columns = ['strikePrice', 'expiryDate', 'impliedVolatility', 'pchangeinOpenInterest', 'changeinOpenInterest', 'openInterest', 'totalTradedVolume', 'change', 'lastPrice']
        pe_columns = ['lastPrice', 'change', 'totalTradedVolume', 'openInterest', 'changeinOpenInterest', 'pchangeinOpenInterest', 'impliedVolatility', 'underlyingValue']

        ce_df = ce_df[ce_columns]
        pe_df = pe_df[pe_columns]

        ce_df.columns = [f'CE_{col}' for col in ce_columns]
        pe_df.columns = [f'PE_{col}' for col in pe_columns]

        timestamp = data['records']['timestamp']
        dt_object = datetime.strptime(timestamp, '%d-%b-%Y %H:%M:%S')
        date = dt_object.date()  # Extract date object
        time_value = dt_object.time()

        # Convert expiryDate to date format
        ce_df['CE_expiryDate'] = pd.to_datetime(ce_df['CE_expiryDate'], format='%d-%b-%Y').dt.date

        df = pd.concat([ce_df, pe_df], axis=1)
        df.insert(0, 'Date', date)
        df.insert(1, 'Time', time_value)
        df.rename(columns={'CE_strikePrice': 'strikePrice', 'CE_expiryDate': 'expiryDate', 'PE_underlyingValue': 'underlyingValue'}, inplace=True)
        df['LowerBound'] = df['underlyingValue'] - 500
        df['UpperBound'] = df['underlyingValue'] + 500
        df = df[(df['strikePrice'] >= df['LowerBound']) & (df['strikePrice'] <= df['UpperBound'])]
        df['Identifier'] = df.apply(lambda row: f"{row['Date']}-{row['Time']}-{row['strikePrice']}", axis=1)

        return df
    except KeyError as e:
        logging.error(f"Key error: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    return None

def save_to_database(df):
    """Save data to the database in a single transaction."""
    try:
        if df.empty:
            logging.warning("DataFrame is empty. No data saved to the database.")
            return

        table_configs = [
            ('option_chain_historical', 'append'),
            ('option_chain', 'replace')
        ]

        with engine.begin() as connection:  # Use a single transaction
            for table_name, if_exists in table_configs:
                df.to_sql(
                    table_name,
                    con=connection,
                    if_exists=if_exists,
                    index=False,
                    dtype={
                        'Date': Date(),
                        'expiryDate': Date()
                    }
                )

        logging.info("Data saved to database successfully.")
    except Exception as e:
        logging.error(f"Database error: {e}")

# Update cookies initially
update_cookies()
cookie_update_interval = 60*60
last_cookie_update = time.time()

async def background_task():
    global last_cookie_update  # Added global declaration
    last_data_timestamp = None  # Track the last processed data timestamp

    while True:
        current_time = time.time()
        if current_time - last_cookie_update >= cookie_update_interval:
            update_cookies()
            last_cookie_update = current_time

        data = fetch_data()
        if data:
            # Check if the data has a new timestamp
            current_timestamp = data['records']['timestamp']
            if current_timestamp != last_data_timestamp:
                last_data_timestamp = current_timestamp  # Update the last processed timestamp
                df = process_data(data)
                if df is not None:
                    save_to_database(df)

        await asyncio.sleep(10)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")