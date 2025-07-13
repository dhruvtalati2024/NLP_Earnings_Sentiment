import re
import yfinance as yf
import pandas as pd

def load_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_market_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    return df

def align_market_data_with_call(market_df, call_date):
    call_dt = pd.to_datetime(call_date)
    next_day = call_dt + pd.Timedelta(days=1)
    try:
        res = market_df.loc[next_day.strftime('%Y-%m-%d')]
        if isinstance(res, pd.DataFrame):
            res = res.iloc[0]
        return res
    except KeyError:
        next_idx = market_df.index.searchsorted(next_day)
        if next_idx < len(market_df):
            res = market_df.iloc[next_idx]
            return res
        else:
            return None
