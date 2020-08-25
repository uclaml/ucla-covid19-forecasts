import numpy as np
import pandas as pd
import json
import argparse
import us

from model import *
from data import *
from rolling_train import *
from util import *
from datetime import timedelta, datetime

parser = argparse.ArgumentParser(description='validation of prediction performance for all states')
parser.add_argument('--END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--VAL_END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--level', default = "state",
                    help='state, nation or county')
parser.add_argument('--state', default = "default",
                    help='state')
parser.add_argument('--nation', default = "default",
                    help='nation')
parser.add_argument('--dataset', default = "NYtimes",
                    help='nytimes')
args = parser.parse_args()
PRED_START_DATE = args.VAL_END_DATE


print(args)
START_nation = {"Brazil": "2020-03-30", "Canada": "2020-03-28", "Mexico": "2020-03-30", \
 "India": "2020-03-28", "Turkey": "2020-03-22", "Russia": "2020-04-01", "Saudi Arabia": "2020-03-28", "US": "2020-03-22", \
 "United Arab Emirates": "2020-04-10", "Qatar": "2020-04-06", "France": "2020-03-20", "Spain": "2020-03-15", \
 "Indonesia":"2020-03-28", "Peru": "2020-04-06", "Chile": "2020-05-08", "Pakistan": "2020-04-01", "Germany":"2020-03-15", "Italy": "2020-03-10", \
 "South Africa": "2020-04-10", "Sweden": "2020-03-25", "United Kingdom": "2020-03-25", "Colombia": "2020-04-03", "Argentina": "2020-04-03", "Bolivia": "2020-04-26", \
 "Ecuador": "2020-03-28", "Iran": "2020-03-15"}


FR_nation = {"Brazil": [0.2,0.02], "Canada": [0.1,0.015], "Mexico": [0.35, 0.015], 
 "India": [0.20, 0.02], "Turkey": [1, 0.04], "Russia": [0.1, 0.022], "Saudi Arabia": [0.2, 0.035], "US": [0.75, 0.024], \
 "United Arab Emirates": [0.07, 0.04], "Qatar": [0.02, 0.05], "France": [0.25, 0.015], "Spain": [0.4, 0.03], \
 "Indonesia": [0.5, 0.024], "Peru": [0.1, 0.013], "Chile": [0.08, 0.025], "Pakistan": [0.16, 0.025], "Germany":[0.05, 0.001], "Italy":[0.35, 0.02], \
 "South Africa": [0.1, 0.026], "Sweden": [0.5, 0.028], "United Kingdom": [0.5, 0.028], "Colombia": [0.17, 0.01], "Argentina": [0.1, 0.012], "Bolivia": [0.2, 0.015], \
 "Ecuador": [0.5, 0.015], "Iran": [0.2, 0.015]}

decay_state = {"Pennsylvania": [0.7, 0.024], "New York": [0.7, 0.042], "Illinois": [0.7, 0.035], "California": [0.5,0.016], "Massachusetts": [0.7,0.026], "New Jersey": [0.7,0.03], \
"Michigan": [0.8,0.035], "Virginia": [0.7,0.034], "Maryland": [0.7,0.024], "Washington": [0.7,0.036], "North Carolina": [0.7,0.018], "Wisconsin": [0.7,0.034], "Texas": [0.6,0.016], \
"New Mexico": [0.7,0.02], "Louisiana": [0.4,0.02], "Arkansas": [0.7,0.02], "Delaware": [0.7,0.03], "Georgia": [0.7,0.015], "Arizona": [0.7,0.02], "Connecticut": [0.7,0.026], "Ohio": [0.7,0.024], \
"Kentucky": [0.7,0.023], "Kansas": [0.7,0.02], "New Hampshire": [0.7,0.014], "Alabama": [0.7,0.024], "Indiana": [0.7,0.03], "South Carolina": [0.7,0.02], "Colorado": [0.7,0.02], "Florida": [0.4,0.016], \
"West Virginia": [0.7,0.022], "Oklahoma": [0.7,0.03], "Mississippi": [0.7,0.026], "Missouri": [0.7,0.02], "Utah": [0.7,0.018], "Alaska": [0.7,0.04], "Hawaii": [0.7,0.04], "Wyoming": [0.7,0.04], "Maine": [0.7,0.025], \
"District of Columbia": [0.7,0.024], "Tennessee": [0.7,0.027], "Idaho": [0.7,0.02], "Oregon": [0.7,0.036], "Rhode Island": [0.7,0.024], "Nevada": [0.5,0.022], "Iowa": [0.7,0.02], "Minnesota": [0.7,0.025], \
"Nebraska": [0.7,0.02], "Montana": [0.5,0.02]}

mid_dates_state = {"Alabama": "2020-06-03", "Arizona": "2020-05-28", "Arkansas": "2020-05-11", "California": "2020-05-30", "Georgia": "2020-06-05",
 "Nevada": "2020-06-01", "Oklahoma": "2020-05-31", "Oregon": "2020-05-29", "Texas": "2020-06-01", "Ohio": "2020-06-09",
     "West Virginia": "2020-06-08", "Florida": "2020-06-01", "South Carolina": "2020-05-25", "Utah": "2020-05-28", "Iowa": "2020-06-20", "Idaho": "2020-06-15",
     "Montana": "2020-05-15", "Minnesota": "2020-06-20"
}



mid_dates_county = {"San Joaquin": "2020-05-26", "Contra Costa": "2020-06-02", "Alameda": "2020-06-03", "Kern": "2020-05-20", \
 "Tulare": "2020-05-30", "Sacramento": "2020-06-02", "Fresno": "2020-06-07", "San Bernardino": "2020-05-25", \
 "Los Angeles": "2020-06-05", "Santa Clara": "2020-05-29", "Orange": "2020-06-12", "Riverside": "2020-05-26", "San Diego": "2020-06-02" \
 
}
mid_dates_nation = {"US": "2020-06-08", "Mexico": "2020-06-05", "India": "2020-06-05", "South Africa": "2020-06-01", "Iran": "2020-05-03", "Bolivia": "2020-05-25"
}

north_cal = ["Santa Clara", "San Mateo", "Alameda", "Contra Costa", "Sacramento", "San Joaquin", "Fresno"]



def get_county_list(cc_limit=200, pop_limit=50000):
    non_county_list = ["Puerto Rico", "American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands"]
    data = NYTimes(level='counties') if args.dataset == "NYtimes" else JHU_US(level='counties')
    with open("data/county_pop.json", 'r') as f:
        County_Pop = json.load(f)
    county_list = []
    for region in County_Pop.keys():
        county, state = region.split("_")
        if County_Pop[region][0]>=pop_limit and not state in non_county_list:        
            train_data = data.get("2020-03-22", args.END_DATE, state, county)
            confirm, death = train_data[0], train_data[1]
            start_date = get_start_date(train_data)
            if len(death) >0 and np.max(death)>5 and np.max(confirm)>cc_limit and start_date < "2020-05-01":
                county_list += [region]

    return county_list



if args.level == "state":
    data = NYTimes(level='states') if args.dataset == "NYtimes" else JHU_US(level='states')
    nonstate_list = ["American Samoa", "Diamond Princess", "Grand Princess", "Virgin Islands"]
    # region_list = [state for state in data.state_list if not state in nonstate_list]
    mid_dates = mid_dates_state
    val_dir = "val_results_state/"
    pred_dir = "pred_results_state/"
    if not args.state == "default":
        region_list = [args.state]
        region_list = ["New York", "California", "New Jersey", "Illinois", "Florida", "Texas", "Georgia", "Arizona"]
        val_dir = "val_results_state/test"

elif args.level == "county":
    state = "California"
    data = NYTimes(level='counties') if args.dataset == "NYtimes" else JHU_US(level='counties')
    # region_list = get_county_list(cc_limit=1000, pop_limit=5000)
    # print("# feasible counties:", len(region_list))
    mid_dates = mid_dates_county
    val_dir = "val_results_county/" 
    pred_dir = "pred_results_county/"

elif args.level == "nation":
    data = JHU_global()
    # region_list = START_nation.keys()
    mid_dates = mid_dates_nation
    with open("data/world_pop.json", 'r') as f:
        Nation_Pop = json.load(f)
    val_dir = "val_results_world/"
    pred_dir = "pred_results_world/"
    if not args.nation == "default":
        region_list = [args.nation]
        val_dir = "val_results_world/test"

json_file_name = val_dir + args.dataset + "_" + "val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
with open(json_file_name, 'r') as f:
    NE0_region = json.load(f)

prediction_range = 100
frame = []
region_list = list(NE0_region.keys())
for region in region_list:
    
    if args.level == "state":
        state = str(region)
        start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state),100)
        mid_dates = mid_dates_state
        if state in mid_dates.keys():
            second_start_date = mid_dates[state]
            reopen_flag = True
        else:
            second_start_date = "2020-06-15" 
            reopen_flag = False

        train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
        full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, PRED_START_DATE, state)]
        if state in decay_state.keys():
            a, decay = decay_state[state][0], decay_state[state][1]
        else:
            a, decay = 0.7, 0.3

        # json_file_name = "val_results_state/" + args.dataset + "_val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
        # with open(json_file_name, 'r') as f:           
        #     NE0_region = json.load(f)
        pop_in = 1/200
        # will rewrite it using json
        
    elif args.level == "county":
        county, state = region.split(", ")
        region = county + ", " + state
        key = county + "_" + state
        start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state, county))

        if state=="California" and county in mid_dates.keys():
            second_start_date = mid_dates[county]
            reopen_flag = True
        elif state in mid_dates_state.keys() and not (state=="Arkansas" or state == "Montana"):
            second_start_date = mid_dates_state[state]
            reopen_flag = True
        else:
            second_start_date = "2020-06-12"
            reopen_flag = False

        train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, args.END_DATE, state, county)]
        full_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, PRED_START_DATE, state, county)]
        if state in decay_state.keys():
            a, decay = decay_state[state][0], decay_state[state][1]
        else:
            a, decay = 0.7, 0.32
        if county in north_cal and state=="California":
            decay = 0.03
        pop_in = 1/300

        
    elif args.level == "nation":
        nation = str(region)

        if nation in mid_dates_nation.keys():
            second_start_date = mid_dates[nation]
            reopen_flag = True
        elif nation == "Turkey":
            second_start_date = "2020-06-07"
            reopen_flag = False
        else:
            second_start_date = "2020-06-12"
            reopen_flag = False
        start_date = START_nation[nation]
        train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, args.END_DATE, nation)]
        full_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, PRED_START_DATE, nation)]
        a, decay = FR_nation[nation] 
        pop_in = 1/400 if nation == "US" else 1/300


    # determine the parameters including pop_in, N and E_0
    mean_increase = 0
    if len(train_data)>1:
        last_confirm, last_fatality = train_data[-1][0], train_data[-1][1]
        daily_confirm = np.diff(last_confirm)
        mean_increase = np.median(daily_confirm[-7:] - daily_confirm[-14:-7])/2 + np.median(daily_confirm[-14:-7] - daily_confirm[-21:-14])/2
        # if mean_increase<1.1:
        #     pop_in = 1/5000
        if not reopen_flag or args.level == "county":
            if np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1:
                pop_in = 1/5000
            elif mean_increase < np.mean(daily_confirm[-7:])/40:
                pop_in = 1/5000
            elif mean_increase > np.mean(daily_confirm[-7:])/10 and np.mean(daily_confirm[-7:])>60:
                pop_in = 1/500
            else:
                pop_in = 1/1000
        if args.level=="state" and reopen_flag and (np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1):
            pop_in = 1/400
        if args.level == "nation" and (region == "France" or region == "Spain" or region == "Germany" or region == "Italy"):
            pop_in = 1/5000
        if not args.level == "nation" and (state == "New York" or state == "New Jersey"):
            pop_in = 1/5000
    print("region: ", region, " start date: ", start_date, " mid date: ", second_start_date,
        " end date: ", args.END_DATE, " Validation end date: ", args.VAL_END_DATE, "mean increase: ", mean_increase, pop_in )   
    N, E_0 = NE0_region[region][0], NE0_region[region][1]
    # print (N, E_0)
    new_sus = 0 if reopen_flag else 0
    if args.level == "state" or args.level == "county":
        bias = 0.025 if reopen_flag or (state=="Louisiana" or state=="Washington" or state == "North Carolina" or state == "Mississippi") else 0.005
        if state == "Arizona" or state == "Alabama" or state == "Florida" or state=="Indiana" or state=="Wisconsin" or state == "Hawaii" or state == "California" or state=="Texas":
            bias = 0.01
        if state == "Arkansas" or state == "Iowa" or state == "Minnesota" or state == "Louisiana" \
         or state == "Nevada" or state == "Kansas" or state=="Kentucky" or state == "Tennessee" or state == "West Virginia":
            bias = 0.05
        if state == "Texas":
            bias = 0.005
    if args.level == "nation":
        bias = 0.01 if reopen_flag else 0.01

    data_confirm, data_fatality = train_data[0][0], train_data[0][1]
    model = Learner_SuEIR(N=N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay, bias=bias)
    init = [N-E_0-data_confirm[0]-data_fatality[0], E_0, data_confirm[0], data_fatality[0]]

    params_all, loss_all = rolling_train(model, init, train_data, new_sus, pop_in=pop_in)

    
    pred_true = rolling_prediction(model, init, params_all, full_data, new_sus, pred_range=prediction_range, pop_in=pop_in, daily_smooth=True)

    confirm = full_data[0][0][0:-1].tolist() + full_data[1][0][0:-1].tolist() + pred_true[0].tolist()

    plt.figure()
    plt.plot(np.diff(np.array(confirm)))
    plt.savefig("figure_"+args.level+"/daily_increase_"+region+".pdf")
    # print(np.diff(np.array(confirm)))
    plt.close()


    print ("region: ", region, " training loss: ",  \
        loss_all, " maximum death cases: ", int(pred_true[1][-1]), " maximum confirmed cases: ", int(pred_true[0][-1])) 

    _, loss_true = rolling_likelihood(model, init, params_all, train_data, new_sus, pop_in=pop_in)
    data_length = [len(data[0]) for data in train_data]

    prediction_list = []
    interval = 0.3
    params = params_all[1]
    while interval >= -0.0001:
        interval -= 0.01
        beta_list = np.asarray([1-interval,1+interval])*params[0]
        gamma_list = np.asarray([1-interval,1+interval])*params[1]
        sigma_list = np.asarray([1-interval,1+interval])*params[2]
        mu_list = np.asarray([1-interval,1+interval])*params[3]
        for beta0 in beta_list:
            for gamma0 in gamma_list:
                for sigma0 in sigma_list:
                    for mu0 in mu_list:
                        temp_param = [params_all[0]] + [np.asarray([beta0,gamma0,sigma0,mu0])]
                        temp_pred=rolling_prediction(model, init, temp_param, full_data, new_sus, pred_range=prediction_range, pop_in=pop_in, daily_smooth=True)

                        _, loss = rolling_likelihood(model, init, temp_param, train_data, new_sus, pop_in=pop_in)
                        if loss < (9.5/data_length[1]*2+loss_true): ###################### 95% tail probability of Chi square (4) distribution
                            prediction_list += [temp_pred]

    A_inv, I_inv, R_inv = [],[],[]

    prediction_list += [pred_true]

    for _pred in prediction_list:
        I_inv += [_pred[0]]
        R_inv += [_pred[1]]
        A_inv += [_pred[2]]

    I_inv=np.asarray(I_inv)
    R_inv=np.asarray(R_inv)
    A_inv=np.asarray(A_inv)
    
    #set the percentiles of upper and lower bounds
    maxI=np.percentile(I_inv,100,axis=0)
    minI=np.percentile(I_inv,0,axis=0)
    maxR=np.percentile(R_inv,100,axis=0)
    minR=np.percentile(R_inv,0,axis=0)
    maxA=np.percentile(A_inv,100,axis=0)
    minA=np.percentile(A_inv,0,axis=0)
    
    # get the median of the curves
    meanI=I_inv[-1,:]
    meanR=R_inv[-1,:]
    meanA=A_inv[-1,:]
    
    diffR, diffI = np.zeros(R_inv.shape), np.zeros(I_inv.shape)
    diffR[:,1:], diffI[:,1:] = np.diff(R_inv), np.diff(I_inv)
    

    diffmR, diffmI = np.zeros(meanR.shape), np.zeros(meanI.shape)

    diffmR[1:] = np.diff(meanR)
    diffmI[1:] = np.diff(meanI)

    difflR = np.percentile(diffR,0,axis=0)
    diffuR = np.percentile(diffR,100,axis=0)

    difflI = np.percentile(diffI,0,axis=0)
    diffuI = np.percentile(diffI,100,axis=0)


    dates = [pd.to_datetime(PRED_START_DATE)+ timedelta(days=i) \
             for i in range(prediction_range)]
    
    # print(len(dates), len(meanI))
    results0 = np.asarray([minI, maxI, minR, maxR, meanI, meanR, diffmR, difflR, diffuR, minA, maxA, meanA, diffmI, difflI, diffuI])
    results0 = np.asarray(results0.T)
    
    pred_data=pd.DataFrame(data=results0, index = dates, columns=["lower_pre_confirm", "upper_pre_confirm", "lower_pre_fata", "upper_pre_fata",'pre_confirm', \
        'pre_fata','pre_fata_daily','lower_pre_fata_daily','upper_pre_fata_daily','lower_pre_act','upper_pre_act', 'pre_act', \
        'pre_confirm_daily','lower_pre_confirm_daily','upper_pre_confirm_daily'])
    
    if args.level == "state" or args.level == "nation":
        pred_data['Region'] = region
    elif args.level == "county":
        pred_data['Region'] = county
        pred_data["State"] = state

    pred_data=pred_data.reset_index().rename(columns={"index": "Date"})
    frame.append(pred_data[pred_data['Date']>=datetime.strptime(PRED_START_DATE,"%Y-%m-%d")])


result = pd.concat(frame)
save_name = pred_dir + "pred_" + args.level + "_END_DATE_" + args.END_DATE + "_PRED_START_DATE_" + PRED_START_DATE + ".csv"
result.to_csv(save_name, index=False)