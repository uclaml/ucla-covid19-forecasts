import numpy as np
import pandas as pd
import json
import argparse
import us

from util import *
from model import *
from data import *
from rolling_train_modified import *
from datetime import timedelta, datetime

parser = argparse.ArgumentParser(description='validation of prediction performance for all states')
parser.add_argument('--END_DATE', default = "default",
                    help='end date for training models')
parser.add_argument('--VAL_END_DATE', default = "default",
                    help='end date for training models')
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
 "India": [0.20, 0.02], "Turkey": [1, 0.04], "Russia": [0.1, 0.022], "Saudi Arabia": [0.2, 0.035], "US": [0.75, 0.025], \
 "United Arab Emirates": [0.07, 0.04], "Qatar": [0.02, 0.05], "France": [0.25, 0.015], "Spain": [0.4, 0.03], \
 "Indonesia": [0.5, 0.024], "Peru": [0.1, 0.013], "Chile": [0.08, 0.025], "Pakistan": [0.16, 0.025], "Germany":[0.05, 0.01], "Italy":[0.35, 0.02], \
 "South Africa": [0.1, 0.026], "Sweden": [0.5, 0.028], "United Kingdom": [0.5, 0.028], "Colombia": [0.17, 0.01], "Argentina": [0.1, 0.012], "Bolivia": [0.2, 0.015], \
 "Ecuador": [0.5, 0.015], "Iran": [0.5, 0.02]}

decay_state = {"Pennsylvania": [0.7, 0.024], "New York": [0.7, 0.042], "Illinois": [0.7, 0.035], "California": [0.5,0.016], "Massachusetts": [0.7,0.026], "New Jersey": [0.7,0.03], \
"Michigan": [0.8,0.035], "Virginia": [0.7,0.034], "Maryland": [0.7,0.024], "Washington": [0.7,0.036], "North Carolina": [0.7,0.018], "Wisconsin": [0.7,0.034], "Texas": [0.3,0.016], \
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
mid_dates_nation = {"US": "2020-06-28", "Mexico": "2020-07-05", "India": "2020-07-30", "South Africa": "2020-06-01", "Brazil": "2020-07-20", \
 "Iran": "2020-05-03", "Bolivia": "2020-05-25", "Indonesia": "2020-07-01", "Italy": "2020-07-01", "Canada": "2020-08-15", "Russia": "2020-08-20", \
 "United Kindom": "2020-07-08", "Spain": "2020-06-28", "France": "2020-06-28", "Argentina": "2020-08-01", "United Kindom": "2020-07-20" 
}

north_cal = ["Santa Clara", "San Mateo", "Alameda", "Contra Costa", "Sacramento", "San Joaquin", "Fresno"]





pred_start_date = "2020-10-18"

write_file_name = pred_start_date+ "-UCLA-SuEIR2.csv"

Sat_list = [(pd.to_datetime("2020-10-24") + timedelta(days=i*7)).strftime("%Y-%m-%d") for i in range(200)]

data = JHU_US(level="states")
nonstate_list = ["American Samoa", "Diamond Princess", "Grand Princess", "Virgin Islands"]
state_list = ["US"]+[state for state in data.state_list if not state in nonstate_list]
# state_list = ["Texas"]


prediction_range = 100
frame = []
# state_list = ["Minnesota"]
for state in state_list:
    if state == "US":
        nation = state
        data = JHU_global()
        region_list = mid_dates_nation.keys()
        mid_dates = mid_dates_nation
        
        
        second_start_date = mid_dates[nation]
        start_date = START_nation[nation]
        train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, args.END_DATE, nation)]
        full_data = [data.get('2020-03-22', second_start_date, nation), data.get(second_start_date, PRED_START_DATE, nation)]
        # if nation=="US":
        # train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, "2020-09-10", nation), data.get("2020-09-10", args.END_DATE, nation)]
        # full_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, "2020-09-10", nation), data.get("2020-09-10", PRED_START_DATE, nation)]

        a, decay = FR_nation[nation]
        reopen_flag = True 

        json_file_name = "val_results_world/test" + "JHU" + "_val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
        with open(json_file_name, 'r') as f:           
            NE0_region = json.load(f)
        N, E_0 = NE0_region[state][0], NE0_region[state][1]
        pop_in = 1/400

    else:
        data = JHU_US(level="states")
        start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state),100)
        mid_dates = mid_dates_state
        if state in mid_dates.keys():
            second_start_date = mid_dates[state]
            reopen_flag = True
        else:
            second_start_date = "2020-06-15" 
            reopen_flag = False

        train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
        full_data = [data.get('2020-03-22', second_start_date, state), data.get(second_start_date, PRED_START_DATE, state)]
        if state in decay_state.keys():
            a, decay = decay_state[state][0], decay_state[state][1]
        else:
            a, decay = 0.7, 0.3

        json_file_name = "val_results_state/" + "JHU" + "_val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
        with open(json_file_name, 'r') as f:           
            NE0_region = json.load(f)
        N, E_0 = NE0_region[state][0], NE0_region[state][1] 
        pop_in = 1/400

    last_confirm, last_fatality = train_data[-1][0], train_data[-1][1]
    daily_confirm = np.diff(last_confirm)
    mean_increase = np.median(daily_confirm[-7:] - daily_confirm[-14:-7])/2 + np.median(daily_confirm[-14:-7] - daily_confirm[-21:-14])/2
    if not reopen_flag:
        if np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1:
            pop_in = 1/5000
        elif mean_increase < np.mean(daily_confirm[-7:])/40:
            pop_in = 1/5000
        elif mean_increase > np.mean(daily_confirm[-7:])/10 and np.mean(daily_confirm[-7:])>60:
            pop_in = 1/500
        else:
            pop_in = 1/1000
    if reopen_flag and (np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1):
        pop_in = 1/500
    if state == "US":
        pop_in = 1/400

    # print (full_data)
    print("state: ", state, " training end date: ", args.END_DATE, " prediction start date: ", PRED_START_DATE, " mid date: ", second_start_date)  

    new_sus = 0 if reopen_flag else 0
    if not state=="US":

        bias = 0.025 if reopen_flag or (state=="Louisiana" or state=="Washington" or state == "North Carolina" or state == "Mississippi") else 0.005
        if state == "Arizona" or state == "Alabama" or state == "Florida" or state=="Indiana" or state=="Wisconsin" or state == "Hawaii" or state == "California" or state=="Texas" or state=="Illinois":
            bias = 0.01
        if state == "Arkansas" or state == "Iowa" or state == "Minnesota" or state == "Louisiana" \
         or state == "Nevada" or state == "Kansas" or state=="Kentucky" or state == "Tennessee" or state == "West Virginia":
            bias = 0.05
    else:
        bias = 0.015
    data_confirm, data_fatality = train_data[0][0], train_data[0][1]
    model = Learner_SuEIR(N=N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay,bias=bias)
    init = [N-E_0-data_confirm[0]-data_fatality[0], E_0, data_confirm[0], data_fatality[0]]
    params_all, loss_all = rolling_train(model, init, train_data, new_sus, pop_in=pop_in)
    loss_true = [NE0_region[state][-2], NE0_region[state][-1]]
    
    pred_true = rolling_prediction(model, init, params_all, full_data, new_sus, pred_range=prediction_range, pop_in=pop_in, daily_smooth=True)

    confirm = full_data[0][0][0:-1].tolist() + full_data[1][0][0:-1].tolist() + pred_true[0].tolist()


    print ("region: ", state, " training loss: ",  \
        loss_all, loss_true, " maximum death cases: ", int(pred_true[1][-1]), " maximum confirmed cases: ", int(pred_true[0][-1]), "popin", pop_in) 

    interval1=np.linspace(0.7,1.0,num=12)
    interval2=np.linspace(1.3,1.0,num=12)
    ####
    params = params_all[1]
    A_inv, I_inv, R_inv=[],[],[]
    prediction_list=[]
    for index_cof in range(12):
        
        beta_list=np.asarray([interval1[index_cof],interval2[index_cof]])*params[0]
        gamma_list=np.asarray([interval1[index_cof],interval2[index_cof]])*params[1]
        sigma_list=np.asarray([interval1[index_cof],interval2[index_cof]])*params[2]
        mu_list=np.asarray([interval1[index_cof],interval2[index_cof]])*params[3]
        param_list=[]
        for beta0 in beta_list:
            for gamma0 in gamma_list:
                for sigma0 in sigma_list:
                    for mu0 in mu_list:
                        temp_param = [params_all[0]] + [np.asarray([beta0,gamma0,sigma0,mu0])]
                        temp_pred=rolling_prediction(model, init, temp_param, full_data, new_sus, pred_range=100, pop_in=pop_in, daily_smooth=True)

                        prediction_list += [temp_pred]

        ############## computing the lower and upper bounds
    prediction_list += [pred_true]


    for _pred in prediction_list:
        I_inv += [_pred[0]]
        R_inv += [_pred[1]]
        A_inv += [_pred[2]]       

    # print(len(R_inv))

    I_inv=np.asarray(I_inv)
    R_inv=np.asarray(R_inv)
    A_inv=np.asarray(A_inv)   

    dates = [(pd.to_datetime(PRED_START_DATE)+ timedelta(days=i)).strftime("%Y-%m-%d") for i in range(prediction_range)]

    case_quantiles = [0.025, 0.100, 0.250, 0.500 ,0.750 ,0.900 ,0.975, "NA"]
    death_quantiles = [0.010, 0.025, 0.050, 0.100 ,0.150 ,0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500  \
            ,0.550, 0.600 ,0.650 ,0.700 ,0.750, 0.800, 0.850 ,0.900 ,0.950, 0.975, 0.990, "NA"]


    #### write wk death
    # pred_start_date = PRED_START_DATE
    for quantile in death_quantiles:
        # write cumulative deaths
        week_inds = [i for i in range(len(dates)) if dates[i] in Sat_list]
        R_inv_wk = R_inv[:,week_inds]
        diff_dates = [str(i+1) + " wk ahead cum death" for i in range(len(week_inds))]
        wk_dates = [Sat_list[i] for i in range(len(week_inds))]
        pred_data = {}
        pred_data["location_name"] = state
        pred_data["forecast_date"] = pred_start_date
        pred_data["target"] = diff_dates
        pred_data["target_end_date"] = wk_dates
        pred_data["location"] = state2fips(state)
        pred_data["type"] = "point" if quantile=="NA" else "quantile"
        pred_data["quantile"] = quantile
        pred_data["value"] = R_inv_wk[-1,:].tolist() if quantile=="NA" else np.percentile(R_inv_wk,quantile*100,axis=0).tolist()
        
        df = pd.DataFrame(pred_data)
        frame.append(df)

        # write incidence deaths
        death_lastsat = full_data[-1][1][-1]
        diffR_wk = np.zeros(R_inv_wk.shape)
        diffR_wk[:, 1:] = np.diff(R_inv_wk)
        diffR_wk[:, 0] = R_inv_wk[:,0]-death_lastsat
        diff_dates = [str(i+1) + " wk ahead inc death" for i in range(len(week_inds))]

        pred_data_inc = pred_data.copy()
        pred_data_inc["target"] = diff_dates
        pred_data_inc["value"] = diffR_wk[-1,:].tolist() if quantile=="NA" else np.percentile(diffR_wk,quantile*100,axis=0).tolist()
        

        df = pd.DataFrame(pred_data_inc)
        frame.append(df)

        ################### generating prediction for weeks

    #### write wk case

    for quantile in case_quantiles:
        # write inc cases
        week_inds = [i for i in range(len(dates)) if dates[i] in Sat_list and i<62]
        I_inv_wk = I_inv[:,week_inds]
        wk_dates = [Sat_list[i] for i in range(len(week_inds))]
        death_lastsat = full_data[-1][0][-1]

        diffI_wk = np.zeros(I_inv_wk.shape)
        diffI_wk[:, 1:] = np.diff(I_inv_wk)
        diffI_wk[:, 0] = I_inv_wk[:,0]-death_lastsat
        # print(death_lastsat, I_inv_wk[:,0])
        diff_dates = [str(i+1) + " wk ahead inc case" for i in range(len(week_inds))]

        pred_data_case = {}
        pred_data_case["location_name"] = state
        pred_data_case["forecast_date"] = pred_start_date
        pred_data_case["target"] = diff_dates
        pred_data_case["target_end_date"] = wk_dates
        pred_data_case["location"] = state2fips(state)
        pred_data_case["type"] = "point" if quantile=="NA" else "quantile"
        pred_data_case["quantile"] = quantile
        pred_data_case["value"] = diffI_wk[-1,:].tolist() if quantile=="NA" else np.percentile(diffI_wk,quantile*100,axis=0).tolist()
        

        df = pd.DataFrame(pred_data_case)
        frame.append(df)

        ################### generating prediction for weeks



results = pd.concat(frame).reset_index(drop=True)
# index1=results[results["target"]=="0 day ahead cum death"].index.tolist()
# index2=results[results["target"]=="0 day ahead inc death"].index.tolist()
# results=results.drop(index1)
# results=results.drop(index2)
results.to_csv(write_file_name, index=False)