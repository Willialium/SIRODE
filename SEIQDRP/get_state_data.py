import pandas as pd


def get_cleaned_data(state, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    data = pd.read_csv('../US_daily_data.csv')
    data = data.loc[data['Province_State'] == state]
    data = data[['Date', 'Confirmed', 'Deaths', 'Recovered']]
    data = data.rename(columns={'Deaths': 'Dead'})
    #data['Recovered'] = data['Recovered'].interpolate(method='linear')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.dropna(subset=['Date'])
    data = data.sort_values(by='Date')
    data = data.reset_index()
    data = data[(data['Date'] >= start) & (data['Date'] <= end)]

    return data
