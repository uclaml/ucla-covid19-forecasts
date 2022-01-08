import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import math


# get county/state level data from jhu covid github
def get_JHU(level):
    length = ((datetime.today()-datetime.strptime("2020-03-10", "%Y-%m-%d")).days)

    if level=="nation":
        df_confirm = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
        df_fata = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
        df_recover = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
        regions = df_confirm["Country/Region"].unique()
        frame = []
        for region in regions:
            
            data = {}
            df_confirm_region, df_fata_region = df_confirm[df_confirm["Country/Region"]==region], df_fata[df_fata["Country/Region"]==region]
            df_recover_region = df_recover[df_recover["Country/Region"]==region]
            data_confirm, data_fata = df_confirm_region.values[:,1:].sum(axis=0), df_fata_region.values[:,1:].sum(axis=0)
            data_recover = df_recover_region.values[:,1:].sum(axis=0)
            
            dates = df_confirm.columns
            data_confirm, data_fata, data_recover = np.asarray(data_confirm)[-length:], np.asarray(data_fata)[-length:], np.asarray(data_recover)[-length:]
            dates = np.asarray(dates)[-length:].tolist()
            dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates]
            data["Country_Region"] = region
            data["ConfirmedCases"] = data_confirm
            data["Fatalities"] = data_fata
            data["Recovered"] = data_recover
            data["Date"] = dates
            
            df = pd.DataFrame(data)
            frame.append(df)
            
        results = pd.concat(frame).reset_index(drop=True)

    elif level == "states":

        df_confirm = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
        df_fata = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

        regions = df_confirm["Province_State"].unique()
        frame = []
        for region in regions:
            data = {}
            df_confirm_region, df_fata_region = df_confirm[df_confirm["Province_State"]==region], df_fata[df_fata["Province_State"]==region]
            data_confirm, data_fata = df_confirm_region.values[:,1:].sum(axis=0), df_fata_region.values[:,1:].sum(axis=0)            
            dates = df_confirm.columns
            data_confirm, data_fata = np.asarray(data_confirm)[-length:], np.asarray(data_fata)[-length:]
            dates = np.asarray(dates)[-length:].tolist()
            dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates]
            data["state"] = region
            data["date"] = dates
            data["cases"] = data_confirm
            data["deaths"] = data_fata
            
            df = pd.DataFrame(data)
            frame.append(df)
            
        results = pd.concat(frame).reset_index(drop=True)


    elif level == "counties":
        df_confirm = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
        df_fata = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
        regions = df_confirm["Province_State"].unique()
        frame = []
        for region in regions:    
            df_confirm_region, df_fata_region = df_confirm[df_confirm["Province_State"]==region], \
            df_fata[df_fata["Province_State"]==region]
            
            counties = df_confirm_region["Admin2"].unique()
            for county in counties:
                data = {}
                if isinstance(county, str):
                    df_confirm_county = df_confirm_region[df_confirm_region["Admin2"]==county]
                    df_fata_county = df_fata_region[df_fata_region["Admin2"]==county]
                    fips = df_confirm_county.FIPS.to_numpy()[0]
                    
                    data_confirm, data_fata = df_confirm_county.values[:,1:].sum(axis=0), df_fata_county.values[:,1:].sum(axis=0)
                    dates = df_confirm.columns
                    data_confirm, data_fata = np.asarray(data_confirm)[-length:], np.asarray(data_fata)[-length:]
                    dates = np.asarray(dates)[-length:].tolist()
                    
                    dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates]
                    if not np.isnan(fips):
                        data["county"] = county
                        data["state"] = region
                        data["date"] = dates
                        data["cases"] = data_confirm
                        data["deaths"] = data_fata
                        data["fips"] = "0"+str(int(fips)) if fips<9999 else str(int(fips))

            
                        df = pd.DataFrame(data)
                        frame.append(df)
        
        results = pd.concat(frame).reset_index(drop=True)
    else:
        return 0

    return results
