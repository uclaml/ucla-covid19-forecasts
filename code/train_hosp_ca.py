import numpy as np
from scipy.optimize import minimize
from model import Learner_SuEIR_ICU
from data import NYTimes, Hospital_CA
from train import loss
from matplotlib import pyplot as plt
import us
from datetime import date, timedelta
from datetime import datetime
from rolling_param import get_rolling_bound_ICU
import pandas as pd

class icu_ca():
    def __init__(self, start= "2020-04-22", end = "2020-06-24", period = 29):
        self.monthly_days = period
        self.starting_date=start
        self.ending_date=end    
        self.pop={
            'Los Angeles': 10098052, 'San Diego': 3302833, 'Orange': 3164182, 'Riverside': 2383286,
            'San Bernardino': 2135413, 'Santa Clara': 1922200, 'Alameda': 1643700, 'Sacramento': 1510023,
            'Contra Costa': 1133247, 'Fresno': 978130, 'Kern': 883053, 'San Francisco': 870044,
            'Ventura': 848112, 'San Mateo': 765935, 'San Joaquin': 732212, 'Stanislaus': 539301,
            'Sonoma': 501317, 'Tulare': 460477, 'Santa Barbara': 443738, 'Solano': 438530,
            'Monterey': 433212, 'Placer': 380077, 'San Luis Obispo': 281455, 'Santa Cruz': 273765,
            'Merced': 269075, 'Marin': 260295, 'Butte': 227075, 'Yolo': 214977, 'El Dorado': 186661,
            'Imperial': 180216, 'Shasta': 179085, 'Madera': 155013, 'Kings': 150075, 'Napa': 140530,
            'Humboldt': 135768, 'Nevada': 99092, 'Sutter': 95872, 'Mendocino': 87422, 'Yuba': 75493,
            'Lake': 64148, 'Tehama': 63373, 'San Benito': 59416, 'Tuolumne': 53932, 'Calaveras': 45235
        }
        
        
    def init_loss_train(self, params):
        beta, gamma, sigma, mu, alpha1, rho1, theta1, alpha2, rho2, theta2, S0, E0, I0 = params  # H0, D0, ICU0 known
        dyn_params = [beta, gamma, sigma, mu, alpha1, rho1, theta1, alpha2, rho2, theta2]
        # init = [S,E,I,R,H,D,ICU]
        init = [S0 * self.N, E0 * self.confirm[0], I0 * self.confirm[0], (1 - I0) * self.confirm[0], self.H0, self.D0, self.ICU0]
        _, _, _, _, pred_confirm, pred_death, pred_hospital, pred_ICU = self.model(
            self.size, dyn_params, init)

        return loss(pred_confirm, self.confirm) + loss(pred_death, self.death) + loss(pred_hospital, self.hospital) + loss(pred_ICU, self.icu)

    def loss_train(self, params):
        _, _, _, _, pred_confirm, pred_death, pred_hospital, pred_ICU = self.model(self.size, params, self.init)
        return loss(pred_confirm, self.confirm) + loss(pred_death, self.death) + loss(pred_hospital, self.hospital) + loss(pred_ICU, self.icu)

    #total days
    def days_between(self, d1, d2):
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)+1

    def partitioning_total(self,start,end):
        self.confirm,self.death,self.hospital,self.icu=self.confirm_total[start:end],self.death_total[start:end],self.hospital_total[start:end],self.icu_total[start:end]

    def hstacking(self):
        self.S_all = np.hstack((self.S_all[:-(self.monthly_days-7)], self.S))
        self.E_all = np.hstack((self.E_all[:-(self.monthly_days-7)], self.E))
        self.I_all = np.hstack((self.I_all[:-(self.monthly_days-7)], self.I))
        self.R_all = np.hstack((self.R_all[:-(self.monthly_days-7)], self.R))
        self.H_all = np.hstack((self.H_all[:-(self.monthly_days-7)], self.pred_hospital))
        self.D_all = np.hstack((self.D_all[:-(self.monthly_days-7)], self.pred_death))
        self.ICU_all = np.hstack((self.ICU_all[:-(self.monthly_days-7)], self.pred_ICU))

    def modeling(self,start,end):
        self.S, self.E, self.I, self.R, self.pred_confirm, self.pred_death, self.pred_hospital, self.pred_ICU = self.model(self.size, self.param, self.init)
        self.partitioning_total(start,end)
        self.size=len(self.confirm)
        self.init = [self.S[7], self.E[7], self.I[7], self.confirm[0] - self.I[7], self.hospital[0], self.death[0], self.icu[0]]
        self.init_point=self.param
        self.optimal = minimize(self.loss_train, self.init_point, method='L-BFGS-B', bounds=self.bounds)
        print(self.optimal.fun)
        self.param=self.optimal.x
    
    def rolling(self):
        self.size=len(self.confirm)
        
        self.init_point = [1e-1, 1e-1, 1e-1, 1e-3, # beta, gamma, sigma, mu
                  2e-1, 1e-1, 1e-2, # alpha1, rho1, theta1
                  2e-1, 1e-1, 1e-2, # alpha2, rho2, theta2
                  1, 1e1, 1] # S0, E0, I0
        self.bounds = [(1e-3, 1), (1e-2, 1), (1e-1, 1), (1e-4, 5e-1),
              (1e-1, 2), (1e-1, 2), (1e-4, 1),
              (1e-1, 2), (1e-1, 2), (1e-4, 1),
              (1e-2, 1), (1, 1e2), (1e-2, 1)]
        
        self.optimal = minimize(self.init_loss_train, self.init_point, method='L-BFGS-B', bounds=self.bounds)
        print(self.optimal.fun)
        self.param = self.optimal.x 
        self.beta, self.gamma, self.sigma, self.mu, self.alpha1, self.rho1, self.theta1, self.alpha2, self.rho2, self.theta2, self.S0, self.E0, self.I0 = self.param
        self.dyn_params = [self.beta, self.gamma, self.sigma, self.mu, self.alpha1, self.rho1, self.theta1, self.alpha2, self.rho2, self.theta2]
        self.init = [self.S0 * self.N, self.E0 * self.confirm[0], self.I0 * self.confirm[0], (1 - self.I0) * self.confirm[0], self.H0, self.D0, self.ICU0]
        self.S, self.E, self.I, self.R, self.pred_confirm, self.pred_death, self.pred_hospital, self.pred_ICU = self.model(self.size, self.dyn_params, self.init)

        self.partitioning_total(0,self.monthly_days)

        self.size=len(self.confirm)

        self.init = [self.S[0], self.E[0], self.I[0], self.confirm[0] - self.I[0], self.hospital[0], self.death[0], self.icu[0]]
        self.init_point = self.param[:10]  # use previous one

        self.bounds = [(1e-2, 1), (5e-2, 1), (1e-5, 2e-1), (1e-3, 5e-1),
              (1e-2, 2), (5e-2, 2), (1e-3, 1),
              (1e-2, 2), (5e-2, 2), (1e-3, 1)]

        self.optimal = minimize(self.loss_train, self.init_point, method='L-BFGS-B', bounds=self.bounds)
        if self.optimal.fun < self.loss_train(self.init_point):
            self.param = self.optimal.x
        else:
            self.param = self.init_point


        self.chunk_num=int(-(-(len(self.confirm_total)-self.monthly_days+1)//7))

        for t in range(1,self.chunk_num):
            self.modeling(t*7,self.monthly_days+t*7)
            if t==1:
                self.S_all = self.S
                self.E_all = self.E
                self.I_all = self.I
                self.R_all = self.R
                self.H_all = self.pred_hospital
                self.D_all = self.pred_death
                self.ICU_all = self.pred_ICU
            else:
                self.hstacking()


        if (len(self.confirm_total)-self.monthly_days)%7!=0:
            self.modeling(7*self.chunk_num,len(self.confirm_total))
            self.hstacking()

        self.S, self.E, self.I, self.R, self.pred_confirm, self.pred_death, self.pred_hospital, self.pred_ICU = self.model(self.size, self.param, self.init)
        self.hstacking()
        self.init = [self.S[-1], self.E[-1], self.I[-1], self.confirm[-1] - self.I[-1], self.hospital[-1], self.death[-1], self.icu[-1]]

    
    def plot_prediction(self):
        # Plotting
        if False:
            fig,a = plt.subplots(2,2)
            a[0][0].plot(self.H_all,'r',label="predict")
            a[0][0].plot(self.hospital_total,'b',label="true")
            a[0][0].set_title('Hospitalization')
            a[0][0].legend(loc="upper right")

            a[0][1].plot(self.ICU_all,'r',label="predict")
            a[0][1].plot(self.icu_total,'b',label="true")
            a[0][1].set_title('ICU')
            a[0][1].legend(loc="upper right")

            a[1][0].plot(self.D_all,'r',label="predict")
            a[1][0].plot(self.death_total,'b',label="true")
            a[1][0].set_title('Death')
            a[1][0].legend(loc="upper right") 

            a[1][1].plot(self.I_all+self.R_all,'r',label="predict")
            a[1][1].plot(self.confirm_total,'b',label="true")
            a[1][1].set_title('Confirm')
            a[1][1].legend(loc="upper right") 

            fig.tight_layout(pad=2.0)
            plt.show()          
        else:
            pass
        
        
    def __call__(self):
        self.state_code = "CA"
        self.d1 = NYTimes(level='counties')
        self.d2 = Hospital_CA()

        self.EXPECTED_DAYS=self.days_between(self.starting_date,self.ending_date)
        
        # for perturbation
        start_date=datetime.strptime(self.ending_date, "%Y-%m-%d")
        self.datearr = [str(start_date + timedelta(days=i))[:10] for i in range(100)]
        self.decay = np.exp(np.linspace(0, -2, 100))
        self.hosp_col_up = []
        self.hosp_col_down = []
        self.hosp_col_value = []
        self.icu_col_up = []
        self.icu_col_down = []
        self.icu_col_value = []
        self.col_date = []
        self.col_region = []
        self.perturb = [0.9, 1, 1.1] 
        self.X = np.vstack([x.flatten() for x in np.meshgrid(*[self.perturb for _ in range(7)], sparse=False)]).T
        X_last = [x[-3:] for x in self.X]
        self.X = np.hstack((self.X, X_last))   
        self.q = np.hstack(([1, 2.5], np.linspace(5, 95, 19), [97.5, 99]))
        
        for cty in self.pop.keys():
            self.N = self.pop[cty]
            self.model = Learner_SuEIR_ICU(self.N)
            self.confirm_total, self.death_total = self.d1.get(self.starting_date, self.ending_date, self.state_code, cty)
            self.hospital_total, self.icu_total = self.d2.get(self.starting_date, self.ending_date, cty)

            if len(self.hospital_total) != self.EXPECTED_DAYS or np.isnan(self.hospital_total).any() or np.isnan(self.confirm_total).any() or np.isnan(self.death_total).any() or np.isnan(self.icu_total).any():
                print(cty, "lacks data")

            self.confirm,self.death,self.hospital,self.icu = self.confirm_total[:self.monthly_days],self.death_total[:self.monthly_days],self.hospital_total[:self.monthly_days],self.icu_total[:self.monthly_days]

            print('=' * 5, cty, '=' * 5)

            self.H0 = self.hospital_total[0]
            self.D0 = self.death_total[0]                                                            
            self.ICU0 = self.icu_total[0]

            self.rolling()

            print("parameters are ", self.param)       

            self.plot_prediction()
            
            # add perturbation
            self.I_H = np.vstack([self.decay * self.model(100, self.param * per, self.init)[-2] for per in self.X])# prediction
            self.I_Hp = np.percentile(self.I_H, self.q, axis=0)
            self.hosp_col_up += list(self.I_Hp[-2])
            self.hosp_col_down += list(self.I_Hp[2])
            self.hosp_col_value += list(self.I_Hp[12])

            self.I_ICU = np.vstack([self.decay * self.model(100, self.param * per, self.init)[-1] for per in self.X])# prediction
            self.I_ICUp = np.percentile(self.I_ICU, self.q, axis=0)
            self.icu_col_up += list(self.I_ICUp[-2])
            self.icu_col_down += list(self.I_ICUp[2])
            self.icu_col_value += list(self.I_ICUp[12])

            self.col_date += self.datearr
            self.col_region += [cty for _ in range(100)]
        
        # generate data frame
        print("#"*5, "Generating dataframe","#"*5)
        df = {'Date': self.col_date, 'lower_pred_icu': self.icu_col_down, 'upper_pred_icu': self.icu_col_up, 'pred_icu': self.icu_col_value, 
              'lower_pred_hospital': self.hosp_col_down, 'upper_pred_hospital': self.hosp_col_up, 'pred_hospital': self.hosp_col_value, 
              'Region': self.col_region}
        pd.DataFrame(df).to_csv('pred_icu_ca_{}-{}.csv'.format(str(start_date.month), str(start_date.day)), index=False)        

print('Done Loading Model')

X=icu_ca('2020-04-22','2020-07-15')
X()