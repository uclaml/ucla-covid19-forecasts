import numpy as np
from scipy.optimize import minimize
from model import Learner_SuEIR, Learner_SuEIR_H
from data import NYTimes, Hospital_US, JHU_global



def loss(pred, target, smoothing=10):
    # print (pred)
    return np.mean((np.log(pred+smoothing) - np.log(target+smoothing))**2)

def train(model, init, prev_params, train_data, reg=0, lag=0):

    data_confirm, data_fatality = train_data[0], train_data[1]
    if len(train_data)==3:
        data_fatality = train_data[1] + train_data[2]
    size = len(data_confirm)
    fatality_perday = np.diff(data_fatality)
    target_ave_fatality_perday = np.median(
        fatality_perday[np.maximum(0, len(fatality_perday)-7):])
    confirm_perday = np.diff(data_confirm)
    target_ave_confirm_perday = np.median(
        confirm_perday[np.maximum(0, len(confirm_perday)-7):])

    def loss_train(params):

        _, _, _, pred_remove, pred_confirm, pred_fatality = model(size, params, init, lag)

        pred_fatality = pred_fatality + data_fatality[0] - pred_fatality[0]
        reg = 0.5
        if len(train_data)==3:
            pred_fatality = pred_remove
            reg = 0
        pred_ave_confirm_perday = np.mean(np.maximum(0, np.diff(pred_confirm)[-7:]))
        pred_ave_fatality_perday = np.mean(np.maximum(0, np.diff(pred_fatality)[-7:]))

        # pick_inds = np.arange(0, len(data_fatality), 7)
        # pick_inds = pick_inds + len(data_fatality)-1 - pick_inds[-1]

        # pred_confirm_wk, pred_fatality_wk, data_confirm_wk, data_fatality_wk  \
        # = pred_confirm[pick_inds], pred_fatality[pick_inds], data_confirm[pick_inds], data_fatality[pick_inds]

        reg_loss = loss(np.array(params[2]), np.array(prev_params[2]), smoothing=0)
        # print(np.diff(data_confirm_wk),np.diff(pred_confirm_wk))
        # return loss((pred_confirm_wk), (data_confirm_wk)) \
        #  + loss((pred_fatality_wk), (data_fatality_wk)) \
        #  + 0.1*loss(np.diff(pred_confirm_wk), np.diff(data_confirm_wk)) \
        #   + 0.1*loss(np.diff(pred_fatality_wk), np.diff(data_fatality_wk)) 
            # + 0.5*loss(pred_ave_confirm_perday, target_ave_confirm_perday) + 0.5 * \
            # loss(pred_ave_fatality_perday, target_ave_fatality_perday) + reg * reg_loss

        return loss(pred_confirm, data_confirm) + 1*loss(pred_fatality, data_fatality) 
        + 1*loss(pred_ave_confirm_perday, target_ave_confirm_perday) + 3 * \
            loss(pred_ave_fatality_perday, target_ave_fatality_perday) 

    optimal = minimize(
        loss_train,
        [0.2, .5e-2, 2.5e-1, 0.01],
        method='L-BFGS-B',
        bounds=[(0.0001, .3), (0.001, 0.3), (0.01, 1), (0.001, 1.)]
    )

    return optimal.x, optimal.fun


def rolling_train(model, init, train_data, new_sus, pop_in=1/500):

    lag = 0
    params_all = []
    loss_all = []
    prev_params = [0.2, .5e-2, 3e-1, 0.01]
    reg = 0
    model.reset()
    N = model.N

    # print (mean_increase, pop_in)
    for _train_data in train_data:
        data_confirm, data_fatality = _train_data[0], _train_data[1]
        params, train_loss = train(model, init, prev_params, _train_data, reg=reg, lag=lag)
        pred_sus, pred_exp, pred_act, pred_remove, _, _ = model(len(data_confirm), params, init, lag=lag)
        # print(params)
        lag += len(data_confirm)-10
        reg = 0

        if len(_train_data)==3:
            true_remove = np.minimum(data_confirm[-1], np.maximum(_train_data[1][-1] + _train_data[2][-1], pred_remove[-1]))
        else:
            true_remove = np.minimum(data_confirm[-1], pred_remove[-1])

        init = [pred_sus[-1], pred_exp[-1], data_confirm[-1]-true_remove, true_remove]
        init[0] = init[0] + new_sus
        model.N += new_sus
        model.pop_in = pop_in
        # print (params, train_loss)
        prev_params = params
        params_all += [params]
        loss_all += [train_loss]
    
    
    init[0] = init[0] - new_sus
    model.reset()
    pred_sus, pred_exp, pred_act, pred_remove, pred_confirm, pred_fatality = model(7, params, init, lag=lag)
    
    # print (pred_remove)
    
    return params_all, loss_all 

def rolling_prediction(model, init, params_all, train_data, new_sus, pred_range, pop_in=1/500, daily_smooth=False):
    lag = 0
    model.reset()
    

    for _train_data, params in zip(train_data, params_all):
        data_confirm, data_fatality = _train_data[0], _train_data[1]
        pred_sus, pred_exp, pred_act, pred_remove, _, _ = model(len(data_confirm), params, init, lag=lag)
        
        if len(_train_data)==3:
            true_remove = np.minimum(data_confirm[-1], np.maximum(_train_data[1][-1] + _train_data[2][-1], pred_remove[-1]))
        else:
            true_remove = np.minimum(data_confirm[-1], pred_remove[-1])

        lag += len(data_confirm)-10
        init = [pred_sus[-1], pred_exp[-1], data_confirm[-1]-true_remove, true_remove]
        init[0] = init[0] + new_sus
        model.N += new_sus
        model.pop_in = pop_in
        
        # print(init)
    # print(model.N)
    # if len(train_data)==1:
    init[0] = init[0] - new_sus
    model.N -= new_sus

    model.bias = 14
    pred_sus, pred_exp, pred_act, pred_remove, pred_confirm, pred_fatality = model(pred_range, params, init, lag=lag)
    pred_fatality = pred_fatality + train_data[-1][1][-1] - pred_fatality[0]

    # print(data_fatality)
    # pred_fatality = pred_remove
    fatality_perday = np.diff(np.asarray(data_fatality))
    ave_fatality_perday = np.mean(fatality_perday[-7:])

    confirm_perday = np.diff(np.asarray(data_confirm))
    ave_confirm_perday = np.mean(confirm_perday[-7:])

    slope_fatality_perday  = np.mean(fatality_perday[-7:] -fatality_perday[-14:-7] )/7
    slope_confirm_perday  = np.mean(confirm_perday[-7:] -confirm_perday[-14:-7] )/7

    smoothing = 1. if daily_smooth else 0



    temp_C_perday = np.diff(pred_confirm.copy())
    slope_temp_C_perday = np.diff(temp_C_perday)
    modified_slope_gap_confirm = (slope_confirm_perday - slope_temp_C_perday[0])*smoothing

    modified_slope_gap_confirm = np.maximum(np.minimum(modified_slope_gap_confirm, ave_confirm_perday/40), -ave_confirm_perday/100)
    slope_temp_C_perday = [slope_temp_C_perday[i] + modified_slope_gap_confirm * np.exp(-0.05*i**2) for i in range(len(slope_temp_C_perday))]
    # print (modified_slope_gap_confirm)
    temp_C_perday = [np.maximum(0, temp_C_perday[0] + np.sum(slope_temp_C_perday[0:i])) for i in range(len(slope_temp_C_perday)+1)]
    # print(np.array(temp_C_perday)[1:7])

    # temp_C_perday = np.diff(pred_confirm)
    modifying_gap_confirm = (ave_confirm_perday - temp_C_perday[0])*smoothing
    temp_C_perday  = [np.maximum(0, temp_C_perday[i] + modifying_gap_confirm * np.exp(-0.1*i)) for i in range(len(temp_C_perday))]
    temp_C =  [pred_confirm[0] + np.sum(temp_C_perday[0:i])  for i in range(len(temp_C_perday)+1)]
    pred_confirm = np.array(temp_C)



    temp_F_perday = np.diff(pred_fatality.copy())
    slope_temp_F_perday = np.diff(temp_F_perday)
    smoothing_slope = 0 if np.max(fatality_perday[-7:])>3*np.median(fatality_perday[-7:]) else 1
        
    # print (smoothing)

    modified_slope_gap_fatality = (slope_fatality_perday - slope_temp_F_perday[0])*smoothing_slope
    modified_slope_gap_fatality = np.maximum(np.minimum(modified_slope_gap_fatality, ave_fatality_perday/10), -ave_fatality_perday/20)
    slope_temp_F_perday = [slope_temp_F_perday[i] + modified_slope_gap_fatality * np.exp(-0.05*i**2) for i in range(len(slope_temp_F_perday))]
    temp_F_perday = [np.maximum(0, temp_F_perday[0] + np.sum(slope_temp_F_perday[0:i])) for i in range(len(slope_temp_F_perday)+1)]


    modifying_gap_fatality = (ave_fatality_perday - temp_F_perday[0])*smoothing
    temp_F_perday  = [np.maximum(0, temp_F_perday[i] + modifying_gap_fatality * np.exp(-0.05*i)) for i in range(len(temp_F_perday))]
    temp_F =  [pred_fatality[0] + np.sum(temp_F_perday[0:i])  for i in range(len(temp_F_perday)+1)]
    pred_fatality = np.array(temp_F)

    model.reset()
    return pred_confirm, pred_fatality, pred_act

def rolling_likelihood(model, init, params_all, train_data, new_sus, pop_in):
    lag = 0
    model.reset()
    loss_all = []
    N = model.N
    for _train_data, params in zip(train_data, params_all):
        data_confirm, data_fatality = _train_data[0], _train_data[1]
        pred_sus, pred_exp, pred_act, pred_remove, pred_confirm, pred_fatality = model(len(data_confirm), params, init, lag=lag)
        pred_fatality = pred_fatality + data_fatality[0] - pred_fatality[0]


        est_perday_confirm, data_perday_confirm = np.diff(pred_confirm), np.diff(data_confirm)
        est_perday_fatality, data_perday_fatality = np.diff(pred_fatality), np.diff(data_fatality)

        loss_all += [np.mean(((est_perday_confirm) - (data_perday_confirm))**2/2/est_perday_confirm**2) \
             + np.mean(((est_perday_fatality) - (data_perday_fatality))**2/2/est_perday_fatality**2)]

        if len(_train_data)==3:
            true_remove = np.minimum(data_confirm[-1], np.maximum(_train_data[1][-1] + _train_data[2][-1], pred_remove[-1]))
        else:
            true_remove = np.minimum(data_confirm[-1], pred_remove[-1])
            
        lag += len(data_confirm)-10
        init = [pred_sus[-1], pred_exp[-1], data_confirm[-1]-true_remove, true_remove]
        init[0] = init[0] + new_sus
        model.N += new_sus
        model.pop_in = pop_in

    model.reset()
    return loss_all




if __name__ == '__main__':


    N = 60000000
    E = N/50

    # N = 6585370
    # E = N/70

    data = JHU_global()
    # data = NYTimes(level='states')
    a, decay = 0.75, 0.033
    # state = "California"

    train_data = [data.get('2020-03-22', '2020-05-28', "US"), data.get('2020-05-28', '2020-06-17', "US")]
    data_confirm, data_fatality = train_data[0][0], train_data[0][1]

    init = [N-E-data_confirm[0]-data_fatality[0],
            E, data_confirm[0], data_fatality[0]]


    model = Learner_SuEIR(N=N, E_0=E, I_0=data_confirm[0], R_0=data_fatality[0], a=a, decay=decay)

    params_all, loss_all = rolling_train(model, init, train_data, new_sus=N/2)
    
    pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, new_sus=N/2, pred_range=7)
    print(np.diff(pred_confirm))
    print (pred_confirm)

    confirm = train_data[0][0][0:-1].tolist() + train_data[1][0][0:-1].tolist() + pred_confirm.tolist()
    plt.figure()
    plt.plot(np.diff(np.array(confirm)))
    plt.savefig("figure/daily_increase.pdf")
    # print(np.diff(np.array(confirm)))
    plt.close()
    print(pred_fatality + train_data[-1][1][-1] - pred_fatality[0])