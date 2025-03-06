import requests
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Date  # Import the Date type from SQLAlchemy
import urllib
import time
import http.cookiejar as cookielib
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        ce_df.columns = [f'CE.{col}' for col in ce_columns]
        pe_df.columns = [f'PE.' + col for col in pe_columns]

        timestamp = data['records']['timestamp']
        dt_object = datetime.strptime(timestamp, '%d-%b-%Y %H:%M:%S')
        date = dt_object.date()  # Extract date object
        time_value = dt_object.time()

        # Convert expiryDate to date format
        ce_df['CE.expiryDate'] = pd.to_datetime(ce_df['CE.expiryDate'], format='%d-%b-%Y').dt.date

        df = pd.concat([ce_df, pe_df], axis=1)
        df.insert(0, 'Date', date)
        df.insert(1, 'Time', time_value)
        df.rename(columns={'CE.strikePrice': 'strikePrice', 'CE.expiryDate': 'expiryDate', 'PE.underlyingValue': 'underlyingValue'}, inplace=True)
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
    try:
        params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};"
                                         "SERVER=FELIX\\OPTIONCHAIN;"
                                         "DATABASE=NiftyData;"
                                         "Trusted_Connection=yes;")
        engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
        
        # Insert data into SQL Server with correct data types
        df.to_sql('option_chain_historical', con=engine, if_exists='append', index=False, dtype={
            'Date': Date(),
            'expiryDate': Date()
        })
        df.to_sql('option_chain', con=engine, if_exists='replace', index=False, dtype={
            'Date': Date(),
            'expiryDate': Date()
        })

        logging.info("Data saved to database successfully.")
    except Exception as e:
        logging.error(f"Database error: {e}")

# Update cookies initially
update_cookies()
cookie_update_interval = 900  # 15 minutes in seconds
last_cookie_update = time.time()

while True:
    current_time = time.time()
    if current_time - last_cookie_update >= cookie_update_interval:
        update_cookies()
        last_cookie_update = current_time

    data = fetch_data()
    if data:
        df = process_data(data)
        if df is not None:
            save_to_database(df)
    
    time.sleep(10)  # Wait for 10 seconds before refreshing
