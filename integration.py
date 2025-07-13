import pandas as pd

def integrate_data(vader, finbert, lm, market):
    data = {**vader, **finbert, **lm}
    if market is not None:
        close_val = market['Close']
        volume_val = market['Volume']
        if hasattr(close_val, 'iloc'):
            close_val = close_val.iloc[0]
        if hasattr(volume_val, 'iloc'):
            volume_val = volume_val.iloc[0]
        data['next_day_close'] = float(close_val) if pd.notnull(close_val) else None
        data['next_day_volume'] = float(volume_val) if pd.notnull(volume_val) else None
    else:
        data['next_day_close'] = None
        data['next_day_volume'] = None
    return pd.DataFrame([data])

