import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import sys
from scipy.optimize import curve_fit
from scipy.stats import lognorm
import json
# According to 'us' -package changelog, DC_STATEHOOD env variable should be set truthy before (FIRST) import
import os
os.environ['DC_STATEHOOD'] = '1'
import us

def func(x, a, b, bias):   
    """! Function used apparently to generate a coefficient. Used to add fluctuation to data. Seemingly not used by the program.
    @param x  an array of numeric data
    @param a  a numeric value used to calculate the coefficient
    @param b  a numeric value used to calculate the coefficient
    @param bias  a numeric value used to calculate the coefficient
    @return  the newly calcluated coefficient
    """
    return a*np.minimum(np.exp(-b*(x+bias)), 1)

def func_root(x, a, b, bias):
    """! Function used apparently to generate a coefficient. Instead of using x and bias, it uses the root of x. Seemingly not used by the program.
    @param x  an array of numeric data
    @param a  a numeric value used to calculate the coefficient
    @param b  a numeric value used to calculate the coefficient
    @param bias  an unused parameter
    @return  the newly calcluated coefficient
    """   
    return a*np.minimum(np.exp(-b*(x**0.5)), 1)

def func_new(x, a, b, bias):
    """! Function used apparently to generate a coefficient. Like func, but adds 0.2 to the value. Seemingly not used by the program.
    @param x  an array of numeric data
    @param a  a numeric value used to calculate the coefficient
    @param b  a numeric value used to calculate the coefficient
    @param bias  a numeric value used to calculate the coefficient
    @return  the newly calcluated coefficient
    """   
    return a*np.minimum(np.exp(-b*(x+bias)), 1) + 0.2

def func_poly(x, a, b, bias):
    """! Function used apparently to generate a coefficient. Calculated differently than in func. Seemingly not used by the program.
    @param x  an array of numeric data
    @param a  a numeric value used to calculate the coefficient
    @param b  a numeric value used to calculate the coefficient
    @param bias  a numeric value used to calculate the coefficient
    @return  the newly calcluated coefficient
    """
    return a*np.minimum((x+bias+1)**(-b), 1)

def func_sin(x, a, b):
    """! Function used apparently to generate a coefficient. Called by add_fluctuation.
    @param x  an array of numeric data
    @param a  a numeric value used to calculate the coefficient
    @param b  a numeric value used to calculate the coefficient
    @return  the newly calcluated coefficient
    """
    return a*np.sin(2*np.pi*(x+b)/7)


def lognorm_ave(x, a=4, b=0.568):
    """! Function used to calculate the average of a log-normal distribution random variable.
    @param x  an array of numeric data. The values themselves are not relevant, only the length of the array.
    @param a  a numeric value used for calculation, by default 4
    @param b  a numeric value used for calculation, by default 0.568
    @return  the average of the lognormal random variable
    """
    l = len(x)
    z = np.linspace(0,20,20)
    lognorm_pdf = np.asarray(lognorm.pdf(z/a, b))
    weights = lognorm_pdf[0:l]/lognorm_pdf[0:l].sum()
    weights = weights[::-1]
    return (x*weights).sum()



def add_fluction(x, a=0.3, b=-0.5):
    """! Function used to add fluctuation to a set of numeric data.
    @param x  an array of numeric data. Function aims to generate a fluctuated version of this.
    @param a  a numeric value used for calculation, by default 0.3
    @param b  a numeric value used for calculation, by default -0.5
    @return  an array of numeric data which has been fluctuated from the original x.
    """
    x_inc = np.diff(x)
    num_weeks = int(len(x_inc)/7)
    data_output = [x[0]]
    for i in range(num_weeks):

        data_start = x[7*i]
        data = x_inc[7*i:7*(i+1)]
        data_mean = np.mean(data)
        _x = np.asarray(range(len(data)))
    
        data_new = data_mean * func_sin(_x, a, b) + data_mean
        a *= 0.7
        data_fluc = [data_start + np.sum(data_new[0:i+1]) for i in range(len(data_new))]
        data_output += data_fluc

    data_output += x[num_weeks*7+1:].tolist()

    return np.asarray(data_output)


def write_val_to_json(params_allregion, write_file_name_all, write_file_name_best, limit=.5e-5):
    """! Function used to write validation results into .json files.
    @param params_allregion  array of validation data for each region validated
    @param write_file_name_all  filename for the file where all data from params_allregion is written
    @param write_file_name_best  filename for the file where the best row of data for each region is written
    @param limit  a value used to increase the limit of validation loss which is still be considered 'good'
    """
    dict_file = json.dumps(params_allregion)
    f = open(write_file_name_all,"w")
    f.write(dict_file)
    f.close()
    params_allregion_best = {}
    confirm = 0
    best_list = []
    for _region in params_allregion.keys():
        params = np.asarray(params_allregion[_region])
        min_val, min_ind = np.min(params[:,2]), np.argmin(params[:,2]) 
        good_inds = np.where(params[:,2]<min_val+limit)[0]
        params = params[good_inds,:]
        pick_ind = 3 # 3: last confirm; 4: last death; 5: maximum daily
        candidates = params[:,pick_ind].tolist()
        print (min_val)
        if min_val>1000:
            best_ind = candidates.index(np.percentile(candidates,10,interpolation='nearest'))
        else:
            best_ind = candidates.index(np.percentile(candidates,75,interpolation='nearest'))
        best_list = params[best_ind, :].tolist()
        params_allregion_best[_region] = best_list
        
    total_confirm, total_death = 0, 0           
    for _region in params_allregion_best:
        print (_region, np.asarray(params_allregion_best[_region])[0:3], 
            " Last CC: ",  params_allregion_best[_region][3], 
            " Last DC: ", params_allregion_best[_region][4],
            " max daily CC: ", params_allregion_best[_region][5],)
        total_confirm += params_allregion_best[_region][3]
        total_death += params_allregion_best[_region][4]
    dict_file = json.dumps(params_allregion_best)

    print ("Total confirmed cases: ", total_confirm, " Total deaths: ", total_death)
    f = open(write_file_name_best,"w")
    f.write(dict_file)
    f.close()






def state2fips(state):
    """! Function used return the Federal Information Processing Standard code of the US state
    @param state  the name of the region for which function is called.
    @return  Either the FIPS of the state that variable state validly named, or "US" if not or if FIPS was not found.
    """
    if not state == "US":
        return us.states.lookup(state).fips
    else:
        return "US"

def get_state_list():
    """! Function used to get a list of US states. Function not called anywhere, and appears to reference things which do not exist.
    @return  A list of states (presumably)
    """
    df_Population = pd.read_csv('us-data/us_population.csv') # US states
    df_Train = pd.read_csv('covid-19-data/us-states.csv')    # What even is this file? Path does not exist. Original guys were smoking something alright.
    df_Area = pd.read_csv('us-data/state-areas.csv')         # As above, does not exist. Guess they thought this can be left as is since function not called anywhere
    del_stat_list=[]
    train_countries = df_Train.state.unique().tolist()
    pop_countries =df_Population.STATE.unique().tolist()
    for country in train_countries:
        if country not in pop_countries:
            del_stat_list.append(country)
    state_list=list(set(df_Train['state'].unique()) - set(del_stat_list))

    return state_list

def get_ca_county_list():
    """! Function used to get a list of the counties of California, USA. Function is not called anywhere, and appears to not work.
    @return  A list of CA counties (presumably)
    """
    df_ca_Population = pd.read_csv('ca-data/ca_population.csv') # Does not exist
    file_name = 'covid-19-data/us-counties.csv'                 # Does not exist
    df_Train = pd.read_csv(file_name)
    df_ca_pop_large = df_ca_Population[df_ca_Population["Population"]>1000000]
    df_ca_san_mateo = df_ca_Population[df_ca_Population["County"]=="San Mateo"]
    df_ca_pop_large = pd.concat([df_ca_san_mateo, df_ca_pop_large])
    _county_list = df_ca_pop_large["County"].unique()

    county_list = [[county, "California"] for county in _county_list]
    return county_list

# def get_large_county_list():
#     file_name = 'covid-19-data/us-counties.csv'
#     df_Train = pd.read_csv(file_name)
#     counties = df_Train["county"].unique()
#     num = 0
#     counties = []
#     states =  df_Train["state"].unique()
#     for state in states:
#         df_state = df_Train[df_Train["state"]==state]
#         for county in df_state["county"].unique():
#             max_cc = df_state[df_state["county"]==county]["cases"].max()
#             df_county = df_state[df_state["county"]==county]
#             min_date = df_county[df_county["cases"]>50].date.min()
#             if max_cc >= 1000 and not county=="Unknown" and min_date<"2020-04-25":
#                 counties += [[county, state]]

#     return counties

def get_county_list():
    """! Function to get a list of all counties in USA.
    @return  list of counties in USA
    """
    non_county_list = ["Puerto Rico", "American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands"]
    with open("data/county_pop.json", 'r') as f: # Appears to have all counties of all states
        County_Pop = json.load(f)
    county_list = []
    for region in County_Pop.keys():
        county, state = region.split("_")
        if County_Pop[region][0]>=50000 and not state in non_county_list:
            county_list += [region]

    return county_list

def get_start_date(data, limit=10):
    """! Function to get the start date of the given dataset
    @param data  the dataset of which the start date is wanted. Seems that when called in validation, data is confirmed cases.
    @param limit  the value which is smaller than any wanted value from the data -dataset. Default is 10.
    """
    ind = np.where(data[0]>limit)[0]
    if len(ind)==0:
        START_DATE = "2020-06-01"
    else:
        ind = float(ind[0])
        START_DATE = (pd.to_datetime("2020-03-22") + timedelta(days=ind)).strftime("%Y-%m-%d")
    return START_DATE

def num2str(x):
    """! Helper function to convert a number to a string.
    @param x  a number which is to be converted to a string
    @return  a string object of the original number
    """
    x = str(x)
    if len(x)==4:
        x += "0"
    if len(x)==3:
        x += "00"
    if len(x)==2:
        x += "000"
    return x



def plotting(pred_data, region):
    """! Function presumably used for plotting the data that the model predicts. Not used anywhere in the code.
    @param pred_data  presumably a dataset of data the model has predicted
    @param region  presumably the name of the region for which prediction was made
    """
    fig = plt.figure()
    ax=plt.plot(pred_data["upper_pre_confirm"],'r-')
    ax=plt.plot(pred_data["lower_pre_confirm"],'g-')
    ax=plt.plot(pred_data["pre_confirm"])
    fig.savefig('figure_confirm/comfirm_'+region) # nice typo, confirms function is irrelevant?
    plt.close()
    fig = plt.figure()
    ax=plt.plot(pred_data["upper_pre_fata"],'r-')
    ax=plt.plot(pred_data["lower_pre_fata"],'g-')
    ax=plt.plot(pred_data["pre_fata"])
    fig.savefig('figure_dead/dead_'+region)
    plt.close()
    fig = plt.figure()
    ax=plt.plot(pred_data["upper_diff_fata"],'r-')
    ax=plt.plot(pred_data["lower_diff_fata"],'g-')
    ax=plt.plot(pred_data["diff_fata"])
    fig.savefig('figure_diff/diff_'+region)
    plt.close()
    fig = plt.figure()
    ax=plt.plot(pred_data["upper_pre_act"],'r-')
    ax=plt.plot(pred_data["lower_pre_act"],'g-')
    ax=plt.plot(pred_data["pre_act"])
    fig.savefig('figure_active/act_'+region)
    plt.close()

