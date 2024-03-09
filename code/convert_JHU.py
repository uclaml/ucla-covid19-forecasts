import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import math


def get_JHU(level):
    """! Get county/state/nation level COVID-19 data from JHU GitHub.
    @param level Specifies the level of data to be retrieved (county/state/nation).
    @return Pandas DataFrame containing COVID-19 information about the specific level or 0 if level value is not valid.
    """
    
    # Calculate the number of days from the specified date.
    length = ((datetime.today()-datetime.strptime("2020-03-10", "%Y-%m-%d")).days)

    if level=="nation":
        #df_confirm = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
        #df_fata = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
        #df_recover = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
        confirm = 'data/jhu_confirmed_global.csv'
        death = 'data/jhu_deaths_global.csv'
        recover = 'data/jhu_recovered_global.csv'
        df_confirm = pd.read_csv(confirm)
        df_fata = pd.read_csv(death)
        df_recover = pd.read_csv(recover)
        regions = df_confirm["Country/Region"].unique()

        # Go through the data of all these nations.
        frame = []
        for region in regions:
            data = {}

            # Get confirmed cases, fatalities and recoveries from the nation.
            df_confirm_region, df_fata_region = df_confirm[df_confirm["Country/Region"]==region], df_fata[df_fata["Country/Region"]==region]
            df_recover_region = df_recover[df_recover["Country/Region"]==region]
            data_confirm, data_fata = df_confirm_region.values[:,1:].sum(axis=0), df_fata_region.values[:,1:].sum(axis=0)
            data_recover = df_recover_region.values[:,1:].sum(axis=0)
            
            # Get dates from the CSV.
            dates = df_confirm.columns

            # Get the confirmed cases, fatalities and recoveries from the nation for the last "length" days.
            data_confirm, data_fata, data_recover = np.asarray(data_confirm)[-length:], np.asarray(data_fata)[-length:], np.asarray(data_recover)[-length:]

            # Get the dates from the nation for the last "length" days and change the date format.
            dates = np.asarray(dates)[-length:].tolist()
            dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates]

            # Add nation's COVID-19 data to Data dictionary.
            data["Country_Region"] = region
            data["ConfirmedCases"] = data_confirm
            data["Fatalities"] = data_fata
            data["Recovered"] = data_recover
            data["Date"] = dates
            
            # Create DataFrame from the nation's Data dictionary and add it to the frame list.
            df = pd.DataFrame(data)
            frame.append(df)
            
        # Combine all DataFrames in the frame list to a single DataFrame.
        results = pd.concat(frame).reset_index(drop=True)

    elif level == "states":

        #df_confirm = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
        #df_fata = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
        df_confirm = pd.read_csv('data/jhu_confirmed_us.csv')
        df_fata = pd.read_csv('data/jhu_deaths_us.csv')

        # Get the name of each state.
        regions = df_confirm["Province_State"].unique()

        # Go through the data of all these states.
        frame = []
        for region in regions:
            data = {}

            # Get confirmed cases and fatalities from the state.
            df_confirm_region, df_fata_region = df_confirm[df_confirm["Province_State"]==region], df_fata[df_fata["Province_State"]==region]
            data_confirm, data_fata = df_confirm_region.values[:,1:].sum(axis=0), df_fata_region.values[:,1:].sum(axis=0)

            # Get dates from the CSV.
            dates = df_confirm.columns

            # Get the confirmed cases and fatalities from the state for the last "length" days.
            data_confirm, data_fata = np.asarray(data_confirm)[-length:], np.asarray(data_fata)[-length:]

            # Get the dates from the state for the last "length" days and change the date format.
            dates = np.asarray(dates)[-length:].tolist()
            dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates]

            # Add state's COVID-19 data to Data dictionary.
            data["state"] = region
            data["date"] = dates
            data["cases"] = data_confirm
            data["deaths"] = data_fata
            
            # Create DataFrame from the nation's Data dictionary and add it to the frame list.
            df = pd.DataFrame(data)
            frame.append(df)
        
        # Combine all DataFrames in the frame list to a single DataFrame.
        results = pd.concat(frame).reset_index(drop=True)


    elif level == "counties":
        #df_confirm = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
        #df_fata = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
        df_confirm = pd.read_csv('data/jhu_confirmed_us.csv')
        df_fata = pd.read_csv('data/jhu_deaths_us.csv')
        regions = df_confirm["Province_State"].unique()

        # Go through the data of all these states' counties.
        frame = []
        for region in regions:
            # Get confirmed cases and fatalities from the state.
            df_confirm_region, df_fata_region = df_confirm[df_confirm["Province_State"]==region], \
            df_fata[df_fata["Province_State"]==region]
            
            # Get the name of each county in the current state.
            counties = df_confirm_region["Admin2"].unique()

            # Go through the data of all these counties.
            for county in counties:
                data = {}


                if isinstance(county, str):
                    # Get confirmed cases and fatalities from the current county.
                    df_confirm_county = df_confirm_region[df_confirm_region["Admin2"]==county]
                    df_fata_county = df_fata_region[df_fata_region["Admin2"]==county]

                    # Get FIPS code for the county.
                    fips = df_confirm_county.FIPS.to_numpy()[0]
                    
                    # Get confirmed cases and fatalities from the county.
                    data_confirm, data_fata = df_confirm_county.values[:,1:].sum(axis=0), df_fata_county.values[:,1:].sum(axis=0)

                    # Get dates from the CSV.
                    dates = df_confirm.columns

                    # Get the confirmed cases and fatalities from the state for the last "length" days.
                    data_confirm, data_fata = np.asarray(data_confirm)[-length:], np.asarray(data_fata)[-length:]

                    # Get the dates from the county for the last "length" days and change the date format.
                    dates = np.asarray(dates)[-length:].tolist()
                    dates = [datetime.strptime(d, "%m/%d/%y").strftime('%Y-%m-%d') for d in dates]

                    # Check if FIPS is not NaN.
                    if not np.isnan(fips):
                        # Add county's COVID-19 data to Data dictionary.
                        data["county"] = county
                        data["state"] = region
                        data["date"] = dates
                        data["cases"] = data_confirm
                        data["deaths"] = data_fata
                        data["fips"] = "0"+str(int(fips)) if fips<9999 else str(int(fips))

                        # Create DataFrame from the nation's Data dictionary and add it to the frame list.
                        df = pd.DataFrame(data)
                        frame.append(df)
        # Combine all DataFrames in the frame list to a single DataFrame.
        results = pd.concat(frame).reset_index(drop=True)

    # If "level" is not valid, return 0.
    else:
        return 0

    # Return the results DataFrame containing COVID-19 information about the specific level.
    return results