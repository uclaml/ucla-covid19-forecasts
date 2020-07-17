import numpy as np
from scipy.optimize import minimize
from model import Learner_SuEIR_ICU, Learner_SuEIR_H
from data import NYTimes, Hospital_US
from train import loss
import us
import pandas as pd
from datetime import date, timedelta, datetime
from rolling_param import get_rolling_bound_ICU, get_rolling_bound_H
import time

start = date(2020, 7, 15)
forcast_date = '2020-07-19'
datearr = [str(start + timedelta(days=i)) for i in range(105)]
decay = np.exp(np.linspace(0, -1, 105))

# for pred_I_us.csv
col_date = []
col_state = []
col_pred = []
col_percent = []
col_target_end = []
col_flips = []
col_type = []
col_for = []

# for pred_icu_us.csv
hosp_col_up = []
hosp_col_down = []
hosp_col_value = []

icu_col_up = []
icu_col_down = []
icu_col_value = []

col_state_icu = []
col_date_icu = []

perturb = [0.9, 1, 1.1]
q = np.hstack(([1, 2.5], np.linspace(5, 95, 19), [97.5, 99]))
UI1 = np.zeros((23, 105))

def init_loss_train(params):
    beta, gamma, sigma, mu, alpha1, rho1, theta1, alpha2, rho2, theta2, S0, E0, I0 = params  # H0, D0, ICU0 known
    dyn_params = [beta, gamma, sigma, mu, alpha1, rho1, theta1, alpha2, rho2, theta2]
    # init = [S,E,I,R,H,D,ICU]
    init = [S0 * N, E0 * confirm[0], I0 * confirm[0], (1 - I0) * confirm[0], H0, D0, ICU0]
    _, _, _, _, pred_confirm, pred_death, pred_hospital, pred_ICU = model(
        size, dyn_params, init)

    return loss(pred_confirm, confirm) + loss(pred_death, death) + loss(pred_hospital, hospital) + loss(pred_ICU, icu)

def init_loss_train_H(params):
    beta, gamma, sigma, mu, alpha1, rho1, theta1, S0, E0, I0 = params  # H0, D0known
    dyn_params = [beta, gamma, sigma, mu, alpha1, rho1, theta1]
    # init = [S,E,I,R,H,D,ICU]
    init = [S0 * N, E0 * confirm[0], I0 * confirm[0], (1 - I0) * confirm[0], H0, D0]
    _, _, _, _, pred_confirm, pred_death, pred_hospital = model(
        size, dyn_params, init)

    return loss(pred_confirm, confirm) + loss(pred_death, death) + loss(pred_hospital, hospital)
    
def loss_train(params):
    _, _, _, _, pred_confirm, pred_death, pred_hospital, pred_ICU = model(size, params, init)
    return loss(pred_confirm, confirm) + loss(pred_death, death) + loss(pred_hospital, hospital) + loss(pred_ICU, icu)

def loss_train_H(params):
    _, _, _, _, pred_confirm, pred_death, pred_hospital= model(size, params, init)
    return loss(pred_confirm, confirm) + loss(pred_death, death) + loss(pred_hospital, hospital)

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)+1

population = pd.read_csv('./data/us_population.csv')

bound1 = np.zeros(10)
bound1 = get_rolling_bound_ICU('2020-07-01', 6585370.5, 54878.0875, 'ca')

bound2 = np.zeros(7)
bound2 = get_rolling_bound_H('2020-07-01', 6585370.5, 54878.0875, 'ca')

for entry in population.to_numpy():
    state_code = us.states.lookup(entry[0]).abbr.lower()
    N = entry[1]
    model = Learner_SuEIR_ICU(N)
    
    model_ind = 0
    
    starting_date='2020-04-22'
    ending_date='2020-07-15'
    EXPECTED_DAYS=days_between(starting_date,ending_date)
    
    confirm_total, death_total = NYTimes(level = 'states').get(starting_date, ending_date, state_code)
    hospital_total, icu_total = Hospital_US(state_code).get(starting_date, ending_date)
    
    print('=' * 5, state_code, '=' * 5)
    
    # decide which model to use based on data completeness
    if len(hospital_total) != EXPECTED_DAYS or np.isnan(icu_total).any():
        print(state_code, "lacks ICU data.")
        model = Learner_SuEIR_H(N)
        model_ind = 1
    
    if len(hospital_total) != EXPECTED_DAYS or np.isnan(hospital_total).any():
        print(state_code, "lacks Hospitalization data.")
        continue
    
    if len(hospital_total) != EXPECTED_DAYS or np.isnan(confirm_total).any() or np.isnan(death_total).any():
        print(state_code, "lacks death or confirmed cases data.")
        continue     
        
###################### ICU model####################  
    if model_ind == 0:
        confirm,death,hospital,icu=confirm_total[:30],death_total[:30],hospital_total[:30],icu_total[:30]
        size=len(confirm)
        H0 = hospital_total[0]
        D0 = death_total[0]                                                            
        ICU0 = icu_total[0]

        init_point = [1e-1, 1e-1, 1e-1, 1e-3, # beta, gamma, sigma, mu
                      2e-1, 1e-1, 1e-2, # alpha1, rho1, theta1
                      2e-1, 1e-1, 1e-2, # alpha2, rho2, theta2
                      1, 1e1, 1] # S0, E0, I0

        bounds = [(1e-3, 1), (1e-2, 1), (1e-1, 1), (1e-3, 5e-1),
                  (1e-1, 2), (1e-1, 2), (1e-4, 1),
                  (1e-1, 2), (1e-1, 2), (1e-4, 1),
                  (1e-2, 1), (1, 1e2), (1e-2, 1)]
        optimal = minimize(init_loss_train, init_point, method='L-BFGS-B', bounds=bounds)
        print(optimal.fun)
        param = optimal.x 
        beta, gamma, sigma, mu, alpha1, rho1, theta1, alpha2, rho2, theta2, S0, E0, I0 = param  # H0, D0, ICU0 known
        dyn_params = [beta, gamma, sigma, mu, alpha1, rho1, theta1, alpha2, rho2, theta2]
        init = [S0 * N, E0 * confirm[0], I0 * confirm[0], (1 - I0) * confirm[0], H0, D0, ICU0] 
        S, E, I, R, pred_confirm, pred_death, pred_hospital, pred_ICU = model(size, dyn_params, init)

        confirm,death,hospital,icu=confirm_total[:30],death_total[:30],hospital_total[:30],icu_total[:30]
        size=len(confirm)

        init = [S[0], E[0], I[0], confirm[0] - I[0], hospital[0], death[0], icu[0]]
        init_point = param[:10]  # use previous one    
        bounds = [(1e-3, 1), (1e-2, 1), (1e-1, 1), (1e-3, 5e-1),
                  (1e-1, 2), (1e-1, 2), (1e-4, 1),
                  (1e-1, 2), (1e-1, 2), (1e-4, 1)]
        optimal = minimize(loss_train, init_point, method='L-BFGS-B', bounds=bounds)
        if optimal.fun < loss_train(init_point):
            param = optimal.x
        else:
            param = init_point    

        S, E, I, R, pred_confirm, pred_death, pred_hospital, pred_ICU = model(size, param, init)
        chunk_num=int(-(-(len(confirm_total)-30)//7))
        
        for t in range(1,chunk_num):
            confirm,death,hospital,icu=confirm_total[t*7:30+t*7],death_total[t*7:30+t*7],hospital_total[t*7:30+t*7],icu_total[t*7:30+t*7]
            size=len(confirm)
            init = [S[7], E[7], I[7], confirm[0] - I[7], hospital[0], death[0], icu[0]]
            init_point=param[:10]
            optimal = minimize(loss_train, init_point, method='L-BFGS-B', bounds=bounds)
            print(optimal.fun)
            param=optimal.x

            if t==1:
                S_all = S
                E_all = E
                I_all = I
                R_all = R
                H_all = pred_hospital
                D_all = pred_death
                ICU_all = pred_ICU
            else:
                S_all = np.hstack((S_all[:-22], S))
                E_all = np.hstack((E_all[:-22], E))
                I_all = np.hstack((I_all[:-22], I))
                R_all = np.hstack((R_all[:-22], R))
                H_all = np.hstack((H_all[:-22], pred_hospital))
                D_all = np.hstack((D_all[:-22], pred_death))
                ICU_all = np.hstack((ICU_all[:-22], pred_ICU))

            S, E, I, R, pred_confirm, pred_death, pred_hospital, pred_ICU = model(size, param, init)
        if (len(confirm_total)-30)%7!=0:
            confirm,death,hospital,icu=confirm_total[7*chunk_num:],death_total[7*chunk_num:],hospital_total[7*chunk_num:],icu_total[7*chunk_num:]

            size=len(confirm)
            init = [S[7], E[7], I[7], confirm[0] - I[7], hospital[0], death[0], icu[0]]
            init_point=param
            optimal = minimize(loss_train, init_point, method='L-BFGS-B', bounds=bounds)
            print(optimal.fun)
            param=optimal.x
            S_all = np.hstack((S_all[:-22], S))
            E_all = np.hstack((E_all[:-22], E))
            I_all = np.hstack((I_all[:-22], I))
            R_all = np.hstack((R_all[:-22], R))
            H_all = np.hstack((H_all[:-22], pred_hospital))
            D_all = np.hstack((D_all[:-22], pred_death))
            ICU_all = np.hstack((ICU_all[:-22], pred_ICU))
            S, E, I, R, pred_confirm, pred_death, pred_hospital, pred_ICU = model(size, param, init)    

        S_all = np.hstack((S_all[:-22], S))
        E_all = np.hstack((E_all[:-22], E))
        I_all = np.hstack((I_all[:-22], I))
        R_all = np.hstack((R_all[:-22], R))
        H_all = np.hstack((H_all[:-22], pred_hospital))
        D_all = np.hstack((D_all[:-22], pred_death))
        ICU_all = np.hstack((ICU_all[:-22], pred_ICU))
        init = [S[-1], E[-1], I[-1], confirm[-1] - I[-1], hospital[-1], death[-1], icu[-1]]

        print("parameters are", param)
        
        alpha1 = param[-6]
        alpha2 = param[-3]
        
        X = np.vstack([x.flatten() for x in np.meshgrid(*[perturb for _ in range(7)], sparse=False)]).T
        X_last = [x[-3:] for x in X]
        X = np.hstack((X, X_last))
       
        # Perturbation on I
        I1 = np.vstack([decay * model(105, param * per, init)[2] for per in X])# prediction
        I1p = alpha1 * np.percentile(I1, q, axis=0)
        
        for j, pred in enumerate(I1p):
            col_percent += ['{:.3f}'.format(0.01 * q[j]) for _ in range(100)]
            col_date += ['{} day ahead inc hosp'.format(n) for n in range(1, 101)]
            col_pred += list(pred[5:])
            col_target_end += datearr[5:]
            col_state += [entry[0] for _ in range(100)]
            col_type += ['quantile' for _ in range(100)]
            col_for += [forcast_date for _ in range(100)]
            col_flips += [us.states.lookup(entry[0]).fips for _ in range(100)]
        UI1 += I1p
        # point
        pred = I1p[12]
        col_percent += ['NA' for _ in range(100)]
        col_date += ['{} day ahead inc hosp'.format(n) for n in range(1, 101)]
        col_pred += list(pred[5:])
        col_target_end += datearr[5:]
        col_state += [entry[0] for _ in range(100)]
        col_type += ['point' for _ in range(100)]
        col_for += [forcast_date for _ in range(100)]
        col_flips += [us.states.lookup(entry[0]).fips for _ in range(100)]
        
        I_H = np.vstack([decay * model(105, param * per, init)[-2] for per in X])# prediction
        I_Hp = np.percentile(I_H, q, axis=0)

        hosp_col_up += list(I_Hp[-2])
        hosp_col_down += list(I_Hp[2])
        hosp_col_value += list(I_Hp[12])

        I_ICU = np.vstack([decay * model(105, param * per, init)[-1] for per in X])# prediction
        I_ICUp = np.percentile(I_ICU, q, axis=0)
        icu_col_up += list(I_ICUp[-2])
        icu_col_down += list(I_ICUp[2])
        icu_col_value += list(I_ICUp[12])

        col_date_icu += datearr
        col_state_icu += [entry[0] for _ in range(105)]

        
###################### End of ICU model####################   

###################### Hospitalization MODEL ####################   
    elif model_ind == 1:
        confirm,death,hospital=confirm_total[:30],death_total[:30],hospital_total[:30]
        size=len(confirm)
        
        H0 = hospital_total[0]
        D0 = death_total[0]                                                            

        init_point = [1e-1, 1e-1, 1e-1, 1e-3, # beta, gamma, sigma, mu
                      2e-1, 1e-1, 1e-2, # alpha1, rho1, theta1
                      1, 1e1, 1] # S0, E0, I0
        bounds = [(1e-3, 1), (1e-2, 1), (1e-2, 2e-1), (1e-3, 5e-1),
              (1e-1, 2), (1e-2, 1), (1e-3, 1e-1),
              (1e-2, 1), (1, 1e2), (1e-3, 1)]

        optimal = minimize(init_loss_train_H, init_point, method='L-BFGS-B', bounds=bounds)
        print(optimal.fun)
        param = optimal.x 
        beta, gamma, sigma, mu, alpha1, rho1, theta1, S0, E0, I0 = param  # H0, D0 known
        dyn_params = [beta, gamma, sigma, mu, alpha1, rho1, theta1]
        init = [S0 * N, E0 * confirm[0], I0 * confirm[0], (1 - I0) * confirm[0], H0, D0] 
        S, E, I, R, pred_confirm, pred_death, pred_hospital = model(size, dyn_params, init)

        confirm,death,hospital=confirm_total[:30],death_total[:30],hospital_total[:30]
        size=len(confirm)

        init = [S[0], E[0], I[0], confirm[0] - I[0], hospital[0], death[0]]
        init_point = param[:7]  # use previous one    
        bounds = [(1e-3, 1), (1e-2, 1), (1e-2, 2e-1), (1e-3, 5e-1),
              (1e-1, 2), (1e-2, 1), (1e-3, 1e-1)]
        optimal = minimize(loss_train_H, init_point, method='L-BFGS-B', bounds=bounds)
        if optimal.fun < loss_train_H(init_point):
            param = optimal.x
        else:
            param = init_point    

        S, E, I, R, pred_confirm, pred_death, pred_hospital = model(size, param, init)
        chunk_num=int(-(-(len(confirm_total)-30)//7))

        for t in range(1,chunk_num):
            confirm,death,hospital=confirm_total[t*7:30+t*7],death_total[t*7:30+t*7],hospital_total[t*7:30+t*7]
            size=len(confirm)
            init = [S[7], E[7], I[7], confirm[0] - I[7], hospital[0], death[0]]
            init_point=param[:7]
            optimal = minimize(loss_train_H, init_point, method='L-BFGS-B', bounds=bounds)
            print(optimal.fun)
            param=optimal.x

            if t==1:
                S_all = S
                E_all = E
                I_all = I
                R_all = R
                H_all = pred_hospital
                D_all = pred_death
            else:
                S_all = np.hstack((S_all[:-22], S))
                E_all = np.hstack((E_all[:-22], E))
                I_all = np.hstack((I_all[:-22], I))
                R_all = np.hstack((R_all[:-22], R))
                H_all = np.hstack((H_all[:-22], pred_hospital))
                D_all = np.hstack((D_all[:-22], pred_death))

            S, E, I, R, pred_confirm, pred_death, pred_hospital = model(size, param, init)
        if (len(confirm_total)-30)%7!=0:
            confirm,death,hospital=confirm_total[7*chunk_num:],death_total[7*chunk_num:],hospital_total[7*chunk_num:]
            size=len(confirm)
            init = [S[7], E[7], I[7], confirm[0] - I[7], hospital[0], death[0]]
            init_point=param
            optimal = minimize(loss_train_H, init_point, method='L-BFGS-B', bounds=bounds)
            print(optimal.fun)
            param=optimal.x
            S_all = np.hstack((S_all[:-22], S))
            E_all = np.hstack((E_all[:-22], E))
            I_all = np.hstack((I_all[:-22], I))
            R_all = np.hstack((R_all[:-22], R))
            H_all = np.hstack((H_all[:-22], pred_hospital))
            D_all = np.hstack((D_all[:-22], pred_death))
            S, E, I, R, pred_confirm, pred_death, pred_hospital = model(size, param, init)    

        S_all = np.hstack((S_all[:-22], S))
        E_all = np.hstack((E_all[:-22], E))
        I_all = np.hstack((I_all[:-22], I))
        R_all = np.hstack((R_all[:-22], R))
        H_all = np.hstack((H_all[:-22], pred_hospital))
        D_all = np.hstack((D_all[:-22], pred_death))
        init = [S[-1], E[-1], I[-1], confirm[-1] - I[-1], hospital[-1], death[-1]]

        print("parameters are", param)

        X = np.vstack([x.flatten() for x in np.meshgrid(*[perturb for _ in range(7)], sparse=False)]).T
        
        alpha1 = param[-3]
        
        # Perturbation on I
        I1 = np.vstack([decay * model(105, param * per, init)[2] for per in X])# prediction
        I1p = alpha1 * np.percentile(I1, q, axis=0)
        
        for j, pred in enumerate(I1p):
            col_percent += ['{:.3f}'.format(0.01 * q[j]) for _ in range(100)]
            col_date += ['{} day ahead inc hosp'.format(n) for n in range(1, 101)]
            col_pred += list(pred[5:])
            col_target_end += datearr[5:]
            col_state += [entry[0] for _ in range(100)]
            col_type += ['quantile' for _ in range(100)]
            col_for += [forcast_date for _ in range(100)]
            col_flips += [us.states.lookup(entry[0]).fips for _ in range(100)]
        UI1 += I1p
        # point
        pred = I1p[12]
        col_percent += ['NA' for _ in range(100)]
        col_date += ['{} day ahead inc hosp'.format(n) for n in range(1, 101)]
        col_pred += list(pred[5:])
        col_target_end += datearr[5:]
        col_state += [entry[0] for _ in range(100)]
        col_type += ['point' for _ in range(100)]
        col_for += [forcast_date for _ in range(100)]
        col_flips += [us.states.lookup(entry[0]).fips for _ in range(100)]
        
        # Perturbation on Hospitalization&ICU
        I_H = np.vstack([decay * model(105, param * per, init)[-2] for per in X])# prediction
        I_Hp = np.percentile(I_H, q, axis=0)
        hosp_col_up += list(I_Hp[-2])
        hosp_col_down += list(I_Hp[2])
        hosp_col_value += list(I_Hp[12])
     
        nan_array = np.full(len(I_Hp[-2]), np.nan)
        icu_col_up += list(nan_array)
        icu_col_down += list(nan_array)
        icu_col_value += list(nan_array)

        col_date_icu += datearr
        col_state_icu += [entry[0] for _ in range(105)]      
        
###################### End of Hospitalization MODEL #################### 
print("Done US States")
print("#"*5, "Generating csv", "#"*5)

# Generate pred_I_us.csv
for j, pred in enumerate(UI1):
    col_percent += ['{:.3f}'.format(0.01 * q[j]) for _ in range(100)]
    col_date += ['{} day ahead inc hosp'.format(n) for n in range(1, 101)]
    col_pred += list(1.05 * pred[5:])
    col_target_end += datearr[5:]
    col_state += ['US' for _ in range(100)]
    col_type += ['quantile' for _ in range(100)]
    col_for += ['2020-07-12' for _ in range(100)]
    col_flips += ['US' for _ in range(100)]
# point
pred = UI1[12]
col_percent += ['NA' for _ in range(100)]
col_date += ['{} day ahead inc hosp'.format(n) for n in range(1, 101)]
col_pred += list(1.05 * pred[5:])
col_target_end += datearr[5:]
col_state += ['US' for _ in range(100)]
col_type += ['point' for _ in range(100)]
col_for += ['2020-07-12' for _ in range(100)]
col_flips += ['US' for _ in range(100)]

print(len(col_state), len(col_type), len(col_percent), len(col_pred), len(col_for), len(col_target_end), len(col_date), len(col_flips))

df = {'location_name': col_state, 'type': col_type, 'quantile': col_percent, 'value': col_pred, 
     'forecast_date': col_for, 'target_end_date': col_target_end, 'target': col_date, 'location': col_flips}
pd.DataFrame(df).to_csv('pred_I_us_{}-{}.csv'.format(str(start.month), str(start.day)), index=False)


print(len(col_date_icu), len(icu_col_down), len(icu_col_up), len(icu_col_value), len(hosp_col_down), len(hosp_col_up), len(hosp_col_value), len(col_state_icu))
# Generate pred_icu_us.csv
df2 = {'Date': col_date_icu, 'lower_pred_icu': icu_col_down, 'upper_pred_icu': icu_col_up, 'pred_icu': icu_col_value, 
      'lower_pred_hospital': hosp_col_down, 'upper_pred_hospital': hosp_col_up, 'pred_hospital': hosp_col_value, 
      'Region': col_state_icu}
pd.DataFrame(df2).to_csv('pred_icu_us_{}-{}.csv'.format(str(start.month), str(start.day)), index=False)    