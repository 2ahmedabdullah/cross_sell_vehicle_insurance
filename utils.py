from numpy import nan
import pandas as pd
import yaml
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import warnings
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb


model_path = './saved_models/'
plot_path = './plots/'

def transformation(df_num, df_cat):
    transform1 = {'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0}
    transform2 = {'Male': 1, 'Female': 0}
    transform3 = {'Yes': 1, 'No': 0}
    
    df_cat['Vehicle_Age'] = df_cat['Vehicle_Age'].replace(transform1)
    df_cat['Gender'] = df_cat['Gender'].replace(transform2)
    df_cat['Vehicle_Damage'] = df_cat['Vehicle_Damage'].replace(transform3)
    df_cat.reset_index(drop=True, inplace=True)

    df_num = np.log(df_num+1)
        
    x = df_num.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_num = pd.DataFrame(x_scaled)
    df = pd.concat([df_cat, df_num], axis=1)

    return df

def prediction(model, x_test):
    y_pred = model.predict(x_test)
    pred1 = [item for sublist in y_pred for item in sublist]
    pred2= np.array(pred1)
    pred3= pred2>0.5
    pred4 = pred3*1
    return pred4, pred2


def num_cat(df, num_var, cat_var):
    df_num = df[num_var]
    df_cat = df[cat_var]
    return df_num, df_cat

def nn(x_train, y_train, x_test, y_test):
    input_dim = 9
    model = Sequential()
    model.add(Dense(8, activation='elu', input_shape=(input_dim,)))
    model.add(Dense(5, activation='elu'))
    model.add(Dense(3, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = Adam())
    
    model1= model.fit(x_train, y_train, batch_size=256, epochs=100, 
                      verbose=1, validation_data=(x_test, y_test))
    
    model.save(model_path+'new_model100.h5')
    #saved_model = load_model(model_path+'new_model100.h5')
    
    plt.plot(model1.history['loss'])
    plt.plot(model1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(plot_path+'model_loss.png')    
    plt.show()
    
    
    y_pred, y_probs = prediction(model, x_test)
        
    print('=============NEURAL NETWORK PREDICTIONS==============')
    print(classification_report(y_test, y_pred))
    auc_score = roc_auc_score(y_test, y_probs)
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', round(auc_score, 2))

    fpr, tpr, thresh = metrics.roc_curve(y_test, y_probs)

    return fpr, tpr, auc_score


def logistic(x_train,y_train,x_test,y_test):
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    yy = clf.predict(x_test)
    
    print('================LOGISTIC REGRESSION PREDICTIONS===========')
    print(classification_report(y_test, yy))  
    auc_score = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
    return fpr, tpr, auc_score


def random_forest(x_train,y_train,x_test,y_test):    
    clf1 = RandomForestClassifier(max_depth=10, random_state=0)
    clf1.fit(x_train,y_train)
    yy= clf1.predict(x_test)
    
    print('==================RANDOM FOREST PREDICTIONS===============')
    print(classification_report(y_test, yy))
    auc_score = roc_auc_score(y_test, clf1.predict_proba(x_test)[:, 1])
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, clf1.predict_proba(x_test)[:, 1])
    return fpr, tpr, auc_score
    


def xgboost_model(x_train,y_train,x_test,y_test):  
    model = XGBClassifier()
    warnings.filterwarnings("ignore")
    model.fit(x_train,y_train) 
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]

    print('=====================XGBOOST PREDICTIONS==================')
    print(classification_report(y_test, predictions))
    auc_score = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    auc_score = round(auc_score, 2)
    print ('AUC SCORE: ', auc_score)
    fpr, tpr, thresh = metrics.roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    return fpr, tpr, auc_score


