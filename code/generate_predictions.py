import numpy as np
import pandas as pd
import json
import argparse
# According to 'us' -package changelog, DC_STATEHOOD env variable should be set truthy before (FIRST) import
import os
os.environ['DC_STATEHOOD'] = '1'
import us

from model import *
from data import *
from rolling_train_modified import *
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
parser.add_argument('--popin', type=float, default = 0,
                    help='popin')
args = parser.parse_args()
PRED_START_DATE = args.VAL_END_DATE


print(args)

# Starting dates for predictions for different countries.
START_nation = {"Brazil": "2020-03-30", "Canada": "2020-03-28", "Mexico": "2020-03-30", \
 "India": "2020-03-28", "Turkey": "2020-03-22", "Russia": "2020-04-01", "Saudi Arabia": "2020-03-28", "US": "2020-03-22", \
 "United Arab Emirates": "2020-04-10", "Qatar": "2020-04-06", "France": "2020-03-20", "Spain": "2020-03-15", \
 "Indonesia":"2020-03-28", "Peru": "2020-04-06", "Chile": "2020-05-08", "Pakistan": "2020-04-01", "Germany":"2020-03-15", "Italy": "2020-03-10", \
 "South Africa": "2020-04-10", "Sweden": "2020-03-25", "United Kingdom": "2020-03-25", "Colombia": "2020-04-03", "Argentina": "2020-04-03", "Bolivia": "2020-04-26", \
 "Ecuador": "2020-03-28", "Iran": "2020-03-15"}

# Decays and a values for different countries.
FR_nation = {"Brazil": [0.2,0.02], "Canada": [0.1,0.015], "Mexico": [0.35, 0.015], 
 "India": [0.20, 0.02], "Turkey": [1, 0.04], "Russia": [0.1, 0.022], "Saudi Arabia": [0.2, 0.035], "US": [0.75, 0.02], \
 "United Arab Emirates": [0.07, 0.04], "Qatar": [0.02, 0.05], "France": [0.25, 0.015], "Spain": [0.4, 0.02], \
 "Indonesia": [0.5, 0.02], "Peru": [0.1, 0.013], "Chile": [0.08, 0.025], "Pakistan": [0.16, 0.025], "Germany":[0.4, 0.1], "Italy":[0.35, 0.02], \
 "South Africa": [0.1, 0.026], "Sweden": [0.5, 0.028], "United Kingdom": [0.5, 0.028], "Colombia": [0.17, 0.01], "Argentina": [0.1, 0.012], "Bolivia": [0.2, 0.015], \
 "Ecuador": [0.5, 0.015], "Iran": [0.5, 0.02]}

# Decays and a values for different US states.
decay_state = {"Pennsylvania": [0.7, 0.024], "New York": [0.7, 0.042], "Illinois": [0.7, 0.035], "California": [0.5,0.016], "Massachusetts": [0.7,0.026], "New Jersey": [0.7,0.03], \
"Michigan": [0.8,0.035], "Virginia": [0.7,0.034], "Maryland": [0.7,0.024], "Washington": [0.7,0.036], "North Carolina": [0.7,0.018], "Wisconsin": [0.7,0.034], "Texas": [0.3,0.016], \
"New Mexico": [0.7,0.02], "Louisiana": [0.4,0.02], "Arkansas": [0.7,0.02], "Delaware": [0.7,0.03], "Georgia": [0.7,0.015], "Arizona": [0.7,0.02], "Connecticut": [0.7,0.026], "Ohio": [0.7,0.024], \
"Kentucky": [0.7,0.023], "Kansas": [0.7,0.02], "New Hampshire": [0.7,0.014], "Alabama": [0.7,0.024], "Indiana": [0.7,0.03], "South Carolina": [0.7,0.02], "Colorado": [0.7,0.02], "Florida": [0.4,0.016], \
"West Virginia": [0.7,0.022], "Oklahoma": [0.7,0.03], "Mississippi": [0.7,0.026], "Missouri": [0.7,0.02], "Utah": [0.7,0.018], "Alaska": [0.7,0.04], "Hawaii": [0.7,0.04], "Wyoming": [0.7,0.04], "Maine": [0.7,0.025], \
"District of Columbia": [0.7,0.024], "Tennessee": [0.7,0.027], "Idaho": [0.7,0.02], "Oregon": [0.7,0.036], "Rhode Island": [0.7,0.024], "Nevada": [0.5,0.022], "Iowa": [0.7,0.02], "Minnesota": [0.7,0.025], \
"Nebraska": [0.7,0.02], "Montana": [0.5,0.02]}

# Middle dates for predictions for different US states.
mid_dates_state = {"Alabama": "2020-06-03", "Arizona": "2020-05-28", "Arkansas": "2020-05-11", "California": "2020-05-30", "Georgia": "2020-06-05",
 "Nevada": "2020-06-01", "Oklahoma": "2020-05-31", "Oregon": "2020-05-29", "Texas": "2020-06-15", "Ohio": "2020-06-09",
     "West Virginia": "2020-06-08", "Florida": "2020-06-01", "South Carolina": "2020-05-25", "Utah": "2020-05-28", "Iowa": "2020-06-20", "Idaho": "2020-06-15",
     "Montana": "2020-06-15", "Minnesota": "2020-06-20", "Illinois": "2020-06-30", "New Jersey": "2020-06-30", "North Carolina": "2020-06-20" , "Maryland":  "2020-06-25",
     "Kentucky": "2020-06-30", "Pennsylvania": "2020-07-01", "Colorado": "2020-06-20", "New York": "2020-06-30", "Alaska": "2020-06-30", "Washington": "2020-06-01"
}

# Resurge middle dates for predictions for different US states.
mid_dates_state_resurge = {"Colorado": "2020-09-10", "California": "2020-09-30", "Florida": "2020-09-20", "Illinois": "2020-09-10", "New York": "2020-09-10", "Texas": "2020-09-15"
}

# Middle dates for predictions for different counties in California.
mid_dates_county = {"San Joaquin": "2020-05-26", "Contra Costa": "2020-06-02", "Alameda": "2020-06-03", "Kern": "2020-05-20", \
 "Tulare": "2020-05-30", "Sacramento": "2020-06-02", "Fresno": "2020-06-07", "San Bernardino": "2020-05-25", \
 "Los Angeles": "2020-06-05", "Santa Clara": "2020-05-29", "Orange": "2020-06-12", "Riverside": "2020-05-26", "San Diego": "2020-06-02" \
}

# Middle dates for predictions for different nations,
mid_dates_nation = {"US": "2020-06-15", "Mexico": "2020-07-05", "India": "2020-07-30", "South Africa": "2020-06-01", "Brazil": "2020-07-20", \
 "Iran": "2020-08-30", "Bolivia": "2020-05-25", "Indonesia": "2020-08-01", "Italy": "2020-07-15", "Canada": "2020-08-15", "Russia": "2020-08-20", \
 "United Kindom": "2020-07-08", "Spain": "2020-07-30", "France": "2020-06-28", "Argentina": "2020-08-01", "United Kindom": "2020-07-20", "Canada": "2020-08-30"
}

# Counties in Northern California. NOT USED ON THIS FILE.
north_cal = ["Santa Clara", "San Mateo", "Alameda", "Contra Costa", "Sacramento", "San Joaquin", "Fresno"]


def get_county_list(cc_limit=200, pop_limit=50000):
    """! Function to get a list of all counties based on specific criteria.
    @param cc_limit Minimum number of confirmed cases for a county to be included.
    @param pop_limit Minimum population required for inclusion.
    @return A list of county/state combinations that meet the criteria (format: County_State).
    """

    # List of US territories that are not included in counties.
    non_county_list = ["Puerto Rico", "American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands"]

    # Create object for counties with data from NYTimes or JHU.
    data = NYTimes(level='counties') if args.dataset == "NYtimes" else JHU_US(level='counties')

    # Load populations of US counties.
    with open("data/county_pop.json", 'r') as f:
        County_Pop = json.load(f)
    
    # Go through counties and add them to county_list if certain statements explained below are true.
    county_list = []
    for region in County_Pop.keys():
        county, state = region.split("_")

        # Get data from counties exceeding the pop_limit given to the function.
        if County_Pop[region][0]>=pop_limit and not state in non_county_list:        
            train_data = data.get("2020-03-22", args.END_DATE, state, county)
            confirm, death = train_data[0], train_data[1]
            start_date = get_start_date(train_data)

            # Add county to list if all of the following statements are true.
                # There have been deaths on more than one day.
                # There are more deaths than five.
                # There are more confirmed cases than the cc_limit given to the function.
                # Start date of data is earlier than 2020-05-01.
            if len(death) >0 and np.max(death)>5 and np.max(confirm)>cc_limit and start_date < "2020-05-01":
                county_list += [region]

    return county_list



if args.level == "state":
    # Create object for states with data from NYTimes or JHU.
    data = NYTimes(level='states') if args.dataset == "NYtimes" else JHU_US(level='states')

    # List of US territories and cruise ships not included in states.
    nonstate_list = ["American Samoa", "Diamond Princess", "Grand Princess", "Virgin Islands"]
    # region_list = [state for state in data.state_list if not state in nonstate_list]

    # Get middle dates for different US states and initialize result directories.
    mid_dates = mid_dates_state
    val_dir = "val_results_state/"
    pred_dir = "pred_results_state/"

    # If data for certain state(s) is queried.
    if not args.state == "default":
        # Changes region_list two times. This is overridden later...
        region_list = [args.state]
        region_list = ["New York", "California"]

        # Changes the val_dir to .../test. This results in validation file starting with "test".
        val_dir = "val_results_state/test"

elif args.level == "county":
    # State is California as middle dates are given to different Californian counties.
    state = "California"

    # Create object for counties with data from NYTimes or JHU.
    data = NYTimes(level='counties') if args.dataset == "NYtimes" else JHU_US(level='counties')

    # Get middle dates for different counties in California and initialize result directories.
    mid_dates = mid_dates_county
    val_dir = "val_results_county/" 
    pred_dir = "pred_results_county/"

elif args.level == "nation":
    # Create object for nations with data from JHU.
    data = JHU_global()
    # region_list = START_nation.keys()

    # Get middle dates for nations and load populations of nations.
    mid_dates = mid_dates_nation
    with open("data/world_pop.json", 'r') as f:
        Nation_Pop = json.load(f)

    # Initialize result directories.
    val_dir = "val_results_world/"
    pred_dir = "pred_results_world/"

    # If data for certain nation(s) is queried.
    if not args.nation == "default":
        # Add these nations to list and change the val_dir to .../test -> validation file starts with "test".
        region_list = [args.nation]
        val_dir = "val_results_world/test"

# Give path/name to validation file.
json_file_name = val_dir + args.dataset + "_" + "val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE
if not os.path.exists(json_file_name):
    json_file_name = val_dir + "JHU" + "_" + "val_params_best_END_DATE_" + args.END_DATE + "_VAL_END_DATE_" + args.VAL_END_DATE



# Open the validation file.
with open(json_file_name, 'r') as f:
    NE0_region = json.load(f)

# Add selected regions to region_list excluding Independence, Arkansas.
prediction_range = 100
frame = []
region_list = list(NE0_region.keys())
region_list = [region for region in region_list if not region == "Independence, Arkansas"]

# Go through selected regions.
for region in region_list:
    
    if args.level == "state":
        state = str(region)

        # Get start and middle dates for the state.
        start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state),100)
        mid_dates = mid_dates_state
        if state in mid_dates.keys():
            second_start_date = mid_dates[state]
            reopen_flag = True

        else:
            second_start_date = "2020-08-30" 
            reopen_flag = False

        # Get data from the state for training and full result.
        train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, args.END_DATE, state)]
        full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, PRED_START_DATE, state)]

        # If state is in mid_dates_state list, include resurged start dates.
        if state in mid_dates.keys():
            # Use resurged start date if state is in mid_dates_state_resurge list. Otherwise, use 2020-09-15.
            resurge_start_date = mid_dates_state_resurge[state] if state in mid_dates_state_resurge.keys() else "2020-09-15"

            train_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
             data.get(resurge_start_date, args.END_DATE, state)]
            full_data = [data.get(start_date, second_start_date, state), data.get(second_start_date, resurge_start_date, state), \
             data.get(resurge_start_date, PRED_START_DATE, state)]

        # Use given decay and a value for the state. Otherwise, use values default values.
        if state in decay_state.keys():
            a, decay = decay_state[state][0], decay_state[state][1]

        else:
            a, decay = 0.7, 0.3

        pop_in = 1/400
        
    elif args.level == "county":
        county, state = region.split(", ")
        region = county + ", " + state
        key = county + "_" + state

        # Get start and middle dates for the county.
        start_date = get_start_date(data.get("2020-03-22", args.END_DATE, state, county))
        if state=="California" and county in mid_dates.keys():
            second_start_date = mid_dates[county]
            reopen_flag = True

        elif state in mid_dates_state.keys():
            second_start_date = mid_dates_state[state]
            reopen_flag = True

        else:
            second_start_date = "2020-08-30"
            reopen_flag = False

        # Get data from the county for training and full result.
        train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, args.END_DATE, state, county)]
        full_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, PRED_START_DATE, state, county)]

        # If county's state is in mid_dates_state list, include resurged start dates.
        if state in mid_dates_state.keys():
            # Use resurged start date if state is in mid_dates_state_resurge list. Otherwise, use 2020-09-15.
            resurge_start_date = mid_dates_state_resurge[state] if state in mid_dates_state_resurge.keys() else "2020-09-15"

            train_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, resurge_start_date, state, county), \
             data.get(resurge_start_date, args.END_DATE, state, county)]
            full_data = [data.get(start_date, second_start_date, state, county), data.get(second_start_date, resurge_start_date, state, county), \
             data.get(resurge_start_date, PRED_START_DATE, state, county)]

        # Use given decay and a value for the county's state. Otherwise, use values default values.
        if state in decay_state.keys():
            a, decay = decay_state[state][0], decay_state[state][1]
            
        else:
            a, decay = 0.7, 0.32

        pop_in = 1/400

        
    elif args.level == "nation":
        nation = str(region)

        # Get start and middle dates for the nation.
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

        # Get data from the nation for training and full result.
        train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, args.END_DATE, nation)]
        full_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, PRED_START_DATE, nation)]

        # If nation is US, use different date for some of the data.
        if nation=="US":
            train_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, "2020-09-15", nation), data.get("2020-09-15", args.END_DATE, nation)]
            full_data = [data.get(start_date, second_start_date, nation), data.get(second_start_date, "2020-09-15", nation), data.get("2020-09-15", PRED_START_DATE, nation)]

        # Use given decay and a value for the nation.
        a, decay = FR_nation[nation] 
        pop_in = 1/400 if nation == "US" else 1/400


    # determine the parameters including pop_in, N and E_0
    mean_increase = 0
    if len(train_data)>1:

        # Get last confimed cases and fatalities from training data.
        last_confirm, last_fatality = train_data[-1][0], train_data[-1][1]

        # Get daily and mean values from confirmed cases.
        daily_confirm = np.diff(last_confirm)
        mean_increase = np.median(daily_confirm[-7:] - daily_confirm[-14:-7])/2 + np.median(daily_confirm[-14:-7] - daily_confirm[-21:-14])/2

        # If county/state/nation is not in middle dates list or if county is being inspected.
        if not reopen_flag or args.level == "county":
            # Evaluate daily confirmed cases and mean increases to use a certain pop_in.
            if np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1:
                pop_in = 1/5000

            elif mean_increase < np.mean(daily_confirm[-7:])/40:
                pop_in = 1/5000

            elif mean_increase > np.mean(daily_confirm[-7:])/10 and np.mean(daily_confirm[-7:])>60:
                pop_in = 1/500

            else:
                pop_in = 1/1000
        
        # If state is being inspected and state is in middle dates list and 
        if args.level=="state" and reopen_flag and (np.mean(daily_confirm[-7:])<12.5 or mean_increase<1.1):
            pop_in = 1/500
            
            if state == "California":
                pop_in = 0.01

        # If nation is Germany, Italy or Canada.
        if args.level == "nation" and (region == "Germany" or region == "Italy" or region=="Canada"):
            pop_in = 1/5000

        # If state is New York.
        if not args.level == "nation" and (state == "New York"):
            pop_in = 1/5000

        # If nation is Iran.
        if args.level == "nation" and (region == "Iran"):
            pop_in = 1/1000  

        # If nation is US.
        if args.level == "nation" and (region == "US"):
            pop_in = 1/400

        # Use given pop_in if it is given to the script.
        if args.popin >0:
            pop_in = args.popin

    print("region: ", region, " start date: ", start_date, " mid date: ", second_start_date,
        " end date: ", args.END_DATE, " Validation end date: ", args.VAL_END_DATE, "mean increase: ", mean_increase, pop_in )   
    N, E_0 = NE0_region[region][0], NE0_region[region][1]

    
    new_sus = 0 if reopen_flag else 0
    if args.level == "state" or args.level == "county":
        # Use 0.025 as bias if state is one listed or if the state or county is included in middle dates list. 
        bias = 0.025 if reopen_flag or (state=="Louisiana" or state=="Washington" or state == "North Carolina" or state == "Mississippi") else 0.005

        # Use 0.01 as bias if state is listed.
        if state == "Arizona" or state == "Alabama" or state == "Florida" or state=="Indiana" or state=="Wisconsin" or state == "Hawaii" or state == "California" or state=="Texas" or state=="Illinois":
            bias = 0.01

        # Use 0.05 as bias if state is listed.
        if state == "Arkansas" or state == "Iowa" or state == "Minnesota" or state == "Louisiana" \
         or state == "Nevada" or state == "Kansas" or state=="Kentucky" or state == "Tennessee" or state == "West Virginia":
            bias = 0.05

    if args.level == "nation":
        # Use 0.02 as bias if nation is listed in middle dates list.
        bias = 0.02 if reopen_flag else 0.01
        
        # Use 0.02 as bias if nation is Germany or US.
        if nation == "Germany":
            bias = 0.02
        if nation == "US":
            bias = 0.02

    # Get confimed cases and fatalities from training data.
    data_confirm, data_fatality = train_data[0][0], train_data[0][1]

    # Create model using Learner_SuEIR.
    model = Learner_SuEIR(N=N, E_0=E_0, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay, bias=bias)
    init = [N-E_0-data_confirm[0]-data_fatality[0], E_0, data_confirm[0], data_fatality[0]]

    # Get params_all list and loss_all.
    params_all, loss_all = rolling_train(model, init, train_data, new_sus, pop_in=pop_in)

    # Get true loss and true prediction.
    loss_true = [NE0_region[region][-2], NE0_region[region][-1]]
    pred_true = rolling_prediction(model, init, params_all, full_data, new_sus, pred_range=prediction_range, pop_in=pop_in, daily_smooth=True)

    confirm = full_data[0][0][0:-1].tolist() + full_data[1][0][0:-1].tolist() + pred_true[0].tolist()

    # Plot results.
    plt.figure()
    plt.plot(np.diff(np.array(confirm)))
    plt.savefig("figure_"+args.level+"/daily_increase_"+region+".pdf")
    plt.close()


    print ("region: ", region, " training loss: ",  \
        loss_all, loss_true," maximum death cases: ", int(pred_true[1][-1]), " maximum confirmed cases: ", int(pred_true[0][-1])) 

    _, loss_true = rolling_likelihood(model, init, params_all, train_data, new_sus, pop_in=pop_in)
    data_length = [len(data[0]) for data in train_data]

    # Add predictions to a list.
    prediction_list = []
    interval = 0.3
    params = params_all[1] if len(params_all)==2 else params_all[2]
    while interval >= -0.0001:
        interval -= 0.01

        # Get beta, gamma, sigma and mu lists.
        beta_list = np.asarray([1-interval,1+interval])*params[0]
        gamma_list = np.asarray([1-interval,1+interval])*params[1]
        sigma_list = np.asarray([1-interval,1+interval])*params[2]
        mu_list = np.asarray([1-interval,1+interval])*params[3]

        # Go through these lists.
        for beta0 in beta_list:
            for gamma0 in gamma_list:
                for sigma0 in sigma_list:
                    for mu0 in mu_list:
                        # Create temporary parameter for temporary prediction.
                        temp_param = [params_all[0]] + [np.asarray([beta0,gamma0,sigma0,mu0])]

                        # Modify temporary parameter if there are 3 parameters in params_all list.
                        if len(params_all)==3:
                            temp_param = [params_all[0]] + [params_all[1]] + [np.asarray([beta0,gamma0,sigma0,mu0])]

                        # Create temporary prediction using rolling_prediction.
                        temp_pred=rolling_prediction(model, init, temp_param, full_data, new_sus, pred_range=prediction_range, pop_in=pop_in, daily_smooth=True)

                        _, loss = rolling_likelihood(model, init, temp_param, train_data, new_sus, pop_in=pop_in)

                        if loss < (9.5/data_length[1]*4+loss_true): ###################### 95% tail probability of Chi square (4) distribution
                            prediction_list += [temp_pred]

    A_inv, I_inv, R_inv = [],[],[]

    # Add true prediction to the prediction list.
    prediction_list += [pred_true]

    # Separate prediction components into lists for infections, recoveries and active cases.
    for _pred in prediction_list:
        I_inv += [_pred[0]]
        R_inv += [_pred[1]]
        A_inv += [_pred[2]]

    # Convert lists into NumPy arrays.
    I_inv=np.asarray(I_inv)
    R_inv=np.asarray(R_inv)
    A_inv=np.asarray(A_inv)
    
    # Set the percentiles of upper and lower bounds.
    maxI=np.percentile(I_inv,100,axis=0)
    minI=np.percentile(I_inv,0,axis=0)
    maxR=np.percentile(R_inv,100,axis=0)
    minR=np.percentile(R_inv,0,axis=0)
    maxA=np.percentile(A_inv,100,axis=0)
    minA=np.percentile(A_inv,0,axis=0)
    
    # Get the median of the curves.
    meanI=np.percentile(I_inv,50,axis=0)
    meanR=np.percentile(R_inv,50,axis=0)
    meanA=np.percentile(A_inv,50,axis=0)
    
    # Get differences between values for each recoveries and infections.
    diffR, diffI = np.zeros(R_inv.shape), np.zeros(I_inv.shape)
    diffR[:,1:], diffI[:,1:] = np.diff(R_inv), np.diff(I_inv)
    

    diffmR, diffmI = np.zeros(meanR.shape), np.zeros(meanI.shape)

    # Calculate the lower and upper bounds for recoveries and infections.
    difflR = np.percentile(diffR,0,axis=0)
    diffuR = np.percentile(diffR,100,axis=0)

    difflI = np.percentile(diffI,0,axis=0)
    diffuI = np.percentile(diffI,100,axis=0)

    diffmR = np.percentile(diffR,50,axis=0)
    diffmI = np.percentile(diffI,50,axis=0)

    # Generate list of prediction dates starting from prediction start date. 
    dates = [pd.to_datetime(PRED_START_DATE)+ timedelta(days=i) \
             for i in range(prediction_range)]
    
    # Combine prediction results into NumPy array and transpose it.
    results0 = np.asarray([minI, maxI, minR, maxR, meanI, meanR, diffmR, difflR, diffuR, minA, maxA, meanA, diffmI, difflI, diffuI])
    results0 = np.asarray(results0.T)
    
    # Create DataFrame for prediction data.
    pred_data=pd.DataFrame(data=results0, index = dates, columns=["lower_pre_confirm", "upper_pre_confirm", "lower_pre_fata", "upper_pre_fata",'pre_confirm', \
        'pre_fata','pre_fata_daily','lower_pre_fata_daily','upper_pre_fata_daily','lower_pre_act','upper_pre_act', 'pre_act', \
        'pre_confirm_daily','lower_pre_confirm_daily','upper_pre_confirm_daily'])
    
    # If state or nation, add it to the prediction data region.
    if args.level == "state" or args.level == "nation":
        pred_data['Region'] = region

    # If county, add state and county to prediction data.
    elif args.level == "county":
        pred_data['Region'] = county
        pred_data["State"] = state

    # Reset index and rename "index" column to "Date".
    pred_data=pred_data.reset_index().rename(columns={"index": "Date"})

    # Add the prediction data to the frame list.
    frame.append(pred_data[pred_data['Date']>=datetime.strptime(PRED_START_DATE,"%Y-%m-%d")])

# Combine all dataframes from frame list to a single DataFrame.
result = pd.concat(frame)

# Create filename for result CSV.
save_name = pred_dir + "pred_" + args.level + "_END_DATE_" + args.END_DATE + "_PRED_START_DATE_" + PRED_START_DATE + ".csv"

# Convert result DataFrame to CSV file.
result.to_csv(save_name, index=False)