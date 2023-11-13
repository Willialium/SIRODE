import os
import pandas as pd

def getN(country):
    df = pd.read_csv('COVID-19/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv')
    df = df.loc[df['Country_Region'] == country]
    N = 0
    for state_pop in df['Population']:
        N += state_pop
    return state_pop


def get_cleaned_country_data(country, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    dfC = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    dfR = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    dfD = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

    dfC = dfC.loc[dfC['Country/Region'] == country]
    dfC = dfC.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

    dfR = dfR.loc[dfR['Country/Region'] == country]
    dfR = dfR.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

    dfD = dfD.loc[dfD['Country/Region'] == country]
    dfD = dfD.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

    df = pd.concat([dfC, dfR, dfD], axis=0, ignore_index=True)
    df = df.transpose()
    df = df.reset_index()
    df.columns = ['Date', 'Confirmed', 'Recovered', 'Dead']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= start) & (df['Date'] <= end)]

    return df


