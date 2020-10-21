import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import json
import argparse
import us

from model import *
from data import *
from rolling_train_modified import *
from util import *
from matplotlib import pyplot as plt

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
parser.add_argument('--county', default = "default",
                    help='county')
parser.add_argument('--dataset', default = "NYtimes",
                    help='nytimes')
parser.add_argument('--popin', type=float, default = 0,
                    help='popin')
args = parser.parse_args()

print(args)
START_nation = {"Brazil": "2020-03-30", "Canada": "2020-03-28", "Mexico": "2020-03-30", \
 "India": "2020-03-28", "Turkey": "2020-03-22", "Russia": "2020-04-01", "Saudi Arabia": "2020-03-28", "US": "2020-03-22", \
 "United Arab Emirates": "2020-04-10", "Qatar": "2020-04-06", "France": "2020-03-20", "Spain": "2020-03-15", \
 "Indonesia":"2020-03-28", "Peru": "2020-04-06", "Chile": "2020-05-08", "Pakistan": "2020-04-01", "Germany":"2020-03-15", "Italy": "2020-03-10", \
 "South Africa": "2020-04-10", "Sweden": "2020-03-25", "United Kingdom": "2020-03-25", "Colombia": "2020-04-03", "Argentina": "2020-04-03", "Bolivia": "2020-04-26", \
 "Ecuador": "2020-03-28", "Iran": "2020-03-15"}


FR_nation = {"Brazil": [0.2,0.02], "Canada": [0.1,0.015], "Mexico": [0.35, 0.015], 
 "India": [0.20, 0.02], "Turkey": [1, 0.04], "Russia": [0.1, 0.022], "Saudi Arabia": [0.2, 0.035], "US": [0.75, 0.02], \
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

# severe_state = ["Florida"]  
    

def validation(model, init, params_all, train_data, val_data,  new_sus, pop_in):
    val_data_confirm, val_data_fatality = val_data[0], val_data[1]
    val_size = len(val_data_confirm)

    pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, new_sus, pred_range=val_size, pop_in=pop_in)
    # print(pred_fatality, val_data_fatality)
    # return 0.5*loss(np.maximum(0, pred_fatality-val_data_fatality[0]), np.maximum(0, val_data_fatality-val_data_fatality[0]), smoothing=10) \
    #  + loss(np.maximum(0, pred_confirm-val_data_confirm[0]), np.maximum(0,val_data_confirm-val_data_confirm[0]), smoothing=10)
    
    return  0.5*loss(pred_confirm, val_data_confirm, smoothing=0.1) + loss(pred_fatality, val_data_fatality, smoothing=0.1)

def get_county_list(cc_limit=200, pop_limit=50000):
    non_county_list = ["Puerto Rico", "American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess", "Grand Princess"]
    data = NYTimes(level='counties') if args.dataset == "NYtimes" else JHU_US(level='counties')
    with open("data/county_pop.json", 'r') as f:
        County_Pop = json.load(f)
    county_list = []
    for region in County_Pop.keys():
        county, state = region.split("_")
        # print(state)
        if County_Pop[region][0]>=pop_limit and not state in non_county_list:        
            train_data = data.get("2020-03-22", args.END_DATE, state, county)
            confirm, death = train_data[0], train_data[1]
            start_date = get_start_date(train_data)
            if len(death) >0 and np.max(death)>=0 and np.max(confirm)>cc_limit and start_date < "2020-05-10":# and not county=="Lassen":
                county_list += [region]

    return county_list



if __name__ == '__main__':
    
    
    # initial the dataloader, get region list 
    # get the directory of output validation files
    if args.level == "state":
        # data = NYTimes(level='states')
        data = NYTimes(level='states') if args.dataset == "NYtimes" else JHU_US(level='states')
        nonstate_list = ["American Samoa", "Diamond Princess", "Grand Princess", "Virgin Islands"]
        region_list = [state for state in data.state_list if not state in nonstate_list]
        # region_list = mid_dates_state.keys()
        # print(data.state_list)
        mid_dates = mid_dates_state
        write_dir = "val_results_state/" + args.dataset + "_" 
        if not args.state == "default":
            region_list = [args.state]  
            # region_list = ["New York", "California", "Illinois", "North Carolina", "Florida", "Texas", "Georgia", "Arizona", "South Carolina", "Alabama"]
            # region_list = ["New York", "California"]
            write_dir = "val_results_state/test" + args.dataset + "_"       
        
    elif args.level == "county":
        state = "California"
        # data = NYTimes(level='counties')
        data = NYTimes(level='counties') if args.dataset == "NYtimes" else JHU_US(level='counties')
        # region_list = mid_dates_county.keys()
        # region_list = ["Cochise_Arizona"]
        mid_dates = mid_dates_county
        with open("data/county_pop.json", 'r') as f:
            County_Pop = json.load(f)       

        if not args.state == "default" and not args.county == "default":
            region_list = [args.county + "_" + args.state] 
            # region_list = ["New York_New York", "Los Angeles_California", "Dallas_Texas"] 
            write_dir = "val_results_county/test" + args.dataset + "_"
        else:
            region_list = get_county_list(cc_limit=1000, pop_limit=10)
            print("# feasible counties:", len(region_list))
            write_dir = "val_results_county/" + args.dataset + "_" 

    elif args.level == "nation":
        data = JHU_global()
        region_list = START_nation.keys()
        mid_dates = mid_dates_nation
        write_dir = "val_results_world/" + args.dataset + "_" 
        if not args.nation == "default":
            region_list = [args.nation] 
            write_dir = "val_results_world/test" + args.dataset + "_" 
        with open("data/world_pop.json", 'r') as f:
            Nation_Pop = json.load(f)
        
    
    params_allregion = {}
   

    for region in region_list:

        # generate training data, validation data
        # get the population
        # get the start date, and second start date
        # get the parameters a and decay
        
        if args.level == "state":
            state = str(region)
            df_Population = pd.read_csv('data/us_population.csv')
            print(state)
            Pop=df_Population[df_Population['STATE']==state]["Population"].to_numpy()[0]
            start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state),100)
            if state in mid_dates.keys():
                second_start_date = mid_dates[state]
                train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
                reopen_flag = True
            else:
                second_start_date = "2020-06-15"
                train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
                reopen_flag = False

            val_data = data.get(args.END_DATE, args.VAL_END_DATE, state)
            if state in decay_state.keys():
                a, decay = decay_state[state][0], decay_state[state][1]
            else:
                a, decay = 0.7, 0.3          
            # will rewrite it using json
            pop_in = 1/400
        elif args.level == "county":
            county, state = region.split("_")
            region = county + ", " + state
            key = county + "_" + state

            Pop=County_Pop[key][0]
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

            if start_date < "2020-05-10":
                train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, args.END_DATE, state, county)]
            else:
                train_data = [data.get(start_date, args.END_DATE, state, county)]
            val_data = data.get(args.END_DATE, args.VAL_END_DATE, state, county)
            if state in decay_state.keys():
                a, decay = decay_state[state][0], decay_state[state][1]
            else:
                a, decay = 0.7, 0.32
            if county in north_cal and state=="California":
                decay = 0.03
            pop_in = 1/300
        elif args.level == "nation":
            nation = str(region)
            Pop = Nation_Pop["United States"] if nation == "US" else Nation_Pop[nation]

            if nation in mid_dates_nation.keys():
                second_start_date = mid_dates[nation]
                reopen_flag = True
            elif nation == "Turkey":
                second_start_date = "2020-06-07"
                reopen_flag = False
            else:
                second_start_date = "2020-06-12"
                reopen_flag = False
            pop_in = 1/400 if nation == "US" else 1/300

            start_date = START_nation[nation]
            train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, args.END_DATE, nation)]
            if nation=="US":
                train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, "2020-09-10", nation), data.get("2020-09-10", args.END_DATE, nation)]
            full_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, "2020-09-10", nation), data.get("2020-09-10", args.END_DATE, nation)]

            val_data = data.get(args.END_DATE, args.VAL_END_DATE, nation)
            a, decay = FR_nation[nation]
            
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
                pop_in = 1/500
            if args.level == "nation" and (region == "Germany" or region == "Italy" or region=="Canada"):
                pop_in = 1/5000
            if not args.level == "nation" and (state == "New York" or state == "New Jersey"):
                pop_in = 1/5000
            if args.level == "nation" and (region == "Iran"):
                pop_in = 1/1000  
            if args.level == "nation" and (region == "US"):
                pop_in = 1/400
            if args.popin >0:
                pop_in = args.popin

        print("region: ", region, " start date: ", start_date, " mid date: ", second_start_date,
            " end date: ", args.END_DATE, " Validation end date: ", args.VAL_END_DATE, "mean increase: ", mean_increase, pop_in )    

        # print(train_data)
        # candidate choices of N and E_0, here r = N/E_0
        Ns = np.asarray([0.2])*Pop
        rs = np.asarray([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 120, 150, 200, 400])
        if args.level == "county":
            rs = np.asarray([30,  40, 50, 60, 70, 80,  90, 100, 120, 150, 200, 400])
            # rs = np.asarray([200])

        if args.level == "nation":

            if region == "South Africa" :
                rs *= 4
            if region == "India" or region == "Qatar":
                rs *= 4
            if region == "Argentina" :
                rs *= 4

        A_inv, I_inv, R_inv, loss_list0, loss_list1, params_list, learner_list, I_list = [],[],[],[],[],[],[],[]
            
        val_log = []
        min_val_loss = 10 #used for finding the minimum validation loss
        for N in Ns:
            for r in rs:
                E_0 = N/r

                # In order to simulate the reopen, we assume at the second stage, there are N new suspectible individuals
                new_sus = 0 if reopen_flag else 0
                if args.level == "state" or args.level == "county":
                    bias = 0.025 if reopen_flag or (state=="Louisiana" or state=="Washington" or state == "North Carolina" or state == "Mississippi") else 0.005
                    if state == "Arizona" or state == "Alabama" or state == "Florida" or state=="Indiana" or state=="Wisconsin" or state == "Hawaii" or state == "California" or state=="Texas" or state=="Illinois":
                        bias = 0.01
                    if state == "Arkansas" or state == "Iowa" or state == "Minnesota" or state == "Louisiana" \
                     or state == "Nevada" or state == "Kansas" or state=="Kentucky" or state == "Tennessee" or state == "West Virginia":
                        bias = 0.05
                if args.level == "nation":
                    bias = 0.01 if reopen_flag else 0.01
                    if nation == "Germany":
                        bias = 0.001
                    if nation == "US":
                        bias = 0.015
                data_confirm, data_fatality = train_data[0][0], train_data[0][1]

                model = Learner_SuEIR(N=N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay, bias=bias)

                # At the initialization we assume that there is not recovered cases.
                init = [N-E_0-data_confirm[0]-data_fatality[0], E_0, data_confirm[0], data_fatality[0]]
                # train the model using the candidate N and E_0, then compute the validation loss
                params_all, loss_all = rolling_train(model, init, train_data, new_sus, pop_in=pop_in)
                val_loss = validation(model, init, params_all, train_data, val_data, new_sus, pop_in=pop_in)

                for params in params_all:
                    beta, gamma, sigma, mu = params
                    # we cannot allow mu>sigma otherwise the model is not valid
                    if mu>sigma:
                        val_loss = 1e6


                # using the model to forecast the fatality and confirmed cases in the next 100 days, 
                # output max_daily, last confirm and last fatality for validation
                pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, new_sus, pop_in=pop_in, pred_range=100, daily_smooth=True)
                max_daily_confirm = np.max(np.diff(pred_confirm))
                pred_confirm_last, pred_fatality_last = pred_confirm[-1], pred_fatality[-1]
                # print(np.diff(pred_fatality))
                # print(sigma/mu)
                #prevent the model from explosion
                if pred_confirm_last >  8*train_data[-1][0][-1] or  np.diff(pred_confirm)[-1]>=np.diff(pred_confirm)[-2]:
                    val_loss = 1e8

                # record the information for validation
                val_log += [[N, E_0] + [val_loss] + [pred_confirm_last] + [pred_fatality_last] + [max_daily_confirm] + loss_all  ]

                # plot the daily inc confirm cases
                confirm = train_data[0][0][0:-1].tolist() + train_data[-1][0][0:-1].tolist() + pred_confirm.tolist()
                true_confirm =  train_data[0][0][0:-1].tolist() + train_data[-1][0][0:-1].tolist() + val_data[0][0:-1].tolist()

                deaths = train_data[0][1][0:-1].tolist() + train_data[-1][1][0:-1].tolist() + pred_fatality.tolist()
                true_deaths =  train_data[0][1][0:-1].tolist() + train_data[-1][1][0:-1].tolist() + val_data[1][0:-1].tolist()
                if val_loss < min_val_loss:
                    plt.figure()
                    plt.plot(confirm)
                    plt.plot(true_confirm)
                    plt.savefig("figure_"+args.level+"/daily_increase.pdf")
                    plt.close()
                    plt.figure()
                    plt.plot(deaths)
                    plt.plot(true_deaths)
                    plt.savefig("figure_"+args.level+"/daily_increase_death.pdf")
                    plt.close()
                min_val_loss = np.minimum(val_loss, min_val_loss)
                # print(val_loss)

        params_allregion[region] = val_log
        print (np.asarray(val_log))
        best_log = np.array(val_log)[np.argmin(np.array(val_log)[:,2]),:]
        print("Best Val loss: ", best_log[2], " Last CC: ", best_log[3], " Last FC: ", best_log[4], " Max inc Confirm: ", best_log[5] )

    # write all validation results into files
    write_file_name_all = write_dir + "val_params_" + "END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
    write_file_name_best = write_dir + "val_params_best_" + "END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE

    write_val_to_json(params_allregion, write_file_name_all, write_file_name_best)


            
        