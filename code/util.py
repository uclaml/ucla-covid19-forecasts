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
import us

def func(x, a, b, bias):   
    return a*np.minimum(np.exp(-b*(x+bias)), 1)
def func_root(x, a, b, bias):   
    return a*np.minimum(np.exp(-b*(x**0.5)), 1)
def func_new(x, a, b, bias):   
    return a*np.minimum(np.exp(-b*(x+bias)), 1) + 0.2
def func_poly(x, a, b, bias):
    return a*np.minimum((x+bias+1)**(-b), 1)

def func_sin(x, a, b):
    return a*np.sin(2*np.pi*(x+b)/7)


def lognorm_ave(x, a=4, b=0.568):
    l = len(x)

    z = np.linspace(0,20,20)
    lognorm_pdf = np.asarray(lognorm.pdf(z/a, b))
    weights = lognorm_pdf[0:l]/lognorm_pdf[0:l].sum()
    weights = weights[::-1]

    return (x*weights).sum()



def add_fluction(x, a=0.3, b=-0.5):
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
    if not state == "US":
        return us.states.lookup(state).fips
    else:
        return "US"

def get_state_list():
    df_Population = pd.read_csv('us-data/us_population.csv')
    df_Train = pd.read_csv('covid-19-data/us-states.csv')
    df_Area = pd.read_csv('us-data/state-areas.csv')
    del_stat_list=[]
    train_countries = df_Train.state.unique().tolist()
    pop_countries =df_Population.STATE.unique().tolist()
    for country in train_countries:
        if country not in pop_countries:
            del_stat_list.append(country)
    state_list=list(set(df_Train['state'].unique()) - set(del_stat_list))

    return state_list

def get_ca_county_list():
    df_ca_Population = pd.read_csv('ca-data/ca_population.csv')
    file_name = 'covid-19-data/us-counties.csv'
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
    non_county_list = ["Puerto Rico", "American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands"]
    with open("data/county_pop.json", 'r') as f:
        County_Pop = json.load(f)
    county_list = []
    for region in County_Pop.keys():
        county, state = region.split("_")
        if County_Pop[region][0]>=50000 and not state in non_county_list:
            county_list += [region]

    return county_list

def get_start_date(data, limit=10):
    ind = np.where(data[0]>limit)[0]
    if len(ind)==0:
        START_DATE = "2020-06-01"
    else:
        ind = float(ind[0])
        START_DATE = (pd.to_datetime("2020-03-22") + timedelta(days=ind)).strftime("%Y-%m-%d")
    return START_DATE

def num2str(x):
    x = str(x)
    if len(x)==4:
        x += "0"
    if len(x)==3:
        x += "00"
    if len(x)==2:
        x += "000"
    return x



def plotting(pred_data, region):
    fig = plt.figure()
    ax=plt.plot(pred_data["upper_pre_confirm"],'r-')
    ax=plt.plot(pred_data["lower_pre_confirm"],'g-')
    ax=plt.plot(pred_data["pre_confirm"])
    fig.savefig('figure_confirm/comfirm_'+region)
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

