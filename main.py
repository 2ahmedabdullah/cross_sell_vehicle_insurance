#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:47:11 2022

@author: abdul
"""

import pickle
from utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #lOAD PICKLE
    x_train = pickle.load(open('x_train.pkl', 'rb'))
    y_train = pickle.load(open('y_train.pkl', 'rb'))
    x_test = pickle.load(open('x_test.pkl', 'rb'))
    y_test = pickle.load(open('y_test.pkl', 'rb'))
    
    xgb = xgboost_model(x_train, y_train, x_test, y_test)
    rf = random_forest(x_train, y_train, x_test, y_test)
    log_reg = logistic(x_train, y_train, x_test, y_test)
    n_n = nn(x_train, y_train, x_test, y_test)
    
    
     #PLOTTING ROC CURVE    
    plt.plot(xgb[0], xgb[1], label="XGB, AUC="+str(xgb[2]))
    plt.plot(rf[0], rf[1], label="RF, AUC="+str(rf[2]) )
    plt.plot(log_reg[0],log_reg[1], label="Log_Reg, AUC="+str(log_reg[2]))
    plt.plot(n_n[0], n_n[1], label="Neural_Network1, AUC="+str(n_n[2]))
    plt.plot(([0,1]), ls='dashed',color='black')
    plt.legend(loc=0)
    plt.xlabel('FPR', fontsize=10)
    plt.ylabel('TPR', fontsize=10)
    plt.savefig(plot_path+'ROC.png')
    plt.show()












