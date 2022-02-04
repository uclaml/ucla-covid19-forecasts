# import request
import pandas as pd
import numpy as np
import datetime
import warnings
import us
from convert_JHU import get_JHU

class Data(object):
    def date_range(self):
        warnings.warn('Data range method does not implement')
        raise NotImplementedError

    def get(self, start_date, end_date):
        warnings.warn('Data get method does not implement')
        raise NotImplementedError


'''
Inherit the Data Object:
class xxx(Data):
    def __init__(self):
        pass
    def range(self, start_date, end_date):
        pass
    def get(self, start_date, end_date):
        pass
'''


class NYTimes(Data):
    def __init__(self, level='states'):
        assert level == 'states' or level == 'counties', 'level must be [states|counties]'
        url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-' + level + '.csv'
        self.table = pd.read_csv(url).drop('fips', axis=1)
        assert not self.table.isnull().values.any(), 'We do not handle nan cases in NYTimes'
        self.level = level
        self.state_list = self.table["state"].unique()

    def date_range(self, state, county=None):
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        tab = tab.sort_values(by='date')
        date = tab['date'].unique()
        return date[0], date[-1]

    def get(self, start_date, end_date, state, county=None):

        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        date = pd.to_datetime(tab['date'])
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        # print(end_date)
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        mask = (date >= start) & (date <= end)
        return tab[mask]['cases'].to_numpy(), tab[mask]['deaths'].to_numpy()

class JHU_US(Data):
    def __init__(self, level='states'):
        assert level == 'states' or level == 'counties', 'level must be [states|counties]'
        # url = 'data/train_' + level + '.csv'
        self.table = get_JHU(level)#.drop('fips', axis=1)
        assert not self.table.isnull().values.any(), 'We do not handle nan cases in NYTimes'
        self.level = level
        self.state_list = self.table["state"].unique()

    def date_range(self, state, county=None):
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        tab = tab.sort_values(by='date')
        date = tab['date'].unique()
        return date[0], date[-1]

    def get(self, start_date, end_date, state, county=None):
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        date = pd.to_datetime(tab['date'])
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        # print(end_date)
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        mask = (date >= start) & (date <= end)
        return tab[mask]['cases'].to_numpy(dtype=float), tab[mask]['deaths'].to_numpy(dtype=float)



class JHU_global(Data):
    def __init__(self):
        confirm = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        recover = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
        confirm_table = pd.read_csv(confirm).drop(['Lat', 'Long'], axis=1)
        death_table = pd.read_csv(death).drop(['Lat', 'Long'], axis=1)
        recover_table = pd.read_csv(recover).drop(['Lat', 'Long'], axis=1)
        self.confirm_table = confirm_table.groupby(
            'Country/Region').sum().transpose()
        self.death_table = death_table.groupby(
            'Country/Region').sum().transpose()
        self.recover_table = recover_table.groupby(
            'Country/Region').sum().transpose()

    def date_range(self, country):
        date = pd.to_datetime(self.confirm_table.index).date
        start = str(date[0])
        end = str(date[-1])
        return start, end

    def get(self, start_date, end_date, country):
        date = pd.to_datetime(self.confirm_table.index)
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        confirm = self.confirm_table[country].loc[(
            date >= start) & (date <= end)]
        death = self.death_table[country].loc[(date >= start) & (date <= end)]
        recover = self.recover_table[country].loc[(
            date >= start) & (date <= end)]
        return confirm.to_numpy(), death.to_numpy(), recover.to_numpy()


class Hospital_CA(Data):
    def __init__(self):
        url = 'https://data.chhs.ca.gov/dataset/6882c390-b2d7-4b9a-aefa-2068cee63e47/resource/6cd8d424-dfaa-4bdd-9410-a3d656e1176e/download/covid19data.csv'
        self.table = pd.read_csv(url)[['Most Recent Date', 'County Name',
                                       'COVID-19 Positive Patients', 'ICU COVID-19 Positive Patients']]

    def date_range(self, region):
        table = self.table[self.table['County Name'] == region]
        dates = pd.to_datetime(table['Most Recent Date']).dt.date.to_numpy()
        return dates[0], dates[-1]

    def get(self, start_date, end_date, region):
        table = self.table[self.table['County Name'] == region]
        dates = pd.to_datetime(table['Most Recent Date'])
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        mask = (dates >= start) & (dates <= end)
        return table[mask]['COVID-19 Positive Patients'].to_numpy(), table[mask]['ICU COVID-19 Positive Patients'].to_numpy()

class Hospital_US(Data):
    def __init__(self, state):
        url = 'https://covidtracking.com/api/v1/states/{}/daily.csv'.format(us.states.lookup(state).abbr.lower())
        table = pd.read_csv(url)[['date', 'hospitalizedCurrently', 'inIcuCurrently']]
        # Here we assume that once there is data, then the data is cumulative
        self.table = table[table.notnull().all(axis=1)]
        # print(self.table)
    
    def date_range(self):
        dates = pd.to_datetime(self.table['date'], format='%Y%m%d').dt.date.to_numpy()
        return dates[-1], dates[0]
    
    def get(self, start_date, end_date):
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        dates = pd.to_datetime(self.table['date'], format='%Y%m%d').dt.date.to_numpy()
        mask = (dates >= start) & (dates <= end)
        masked = self.table[mask].sort_values(by='date')
        return masked['hospitalizedCurrently'].to_numpy(), masked['inIcuCurrently'].to_numpy()
        


if __name__ == '__main__':
    data = Hospital_US('california')
    a, b = data.get('2020-04-01', '2020-04-02')
    pass
