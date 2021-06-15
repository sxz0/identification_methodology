import os
import statistics

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
import umap
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)



n_devices=5
datasets=os.listdir("datatests")
n_samples_device = 800 #10000

f = "feat/stability_test_4min_2.csv"#stability_test_loren_long.csv"
datasets=[f]
window=5
for f in datasets:
    df=pd.read_csv("gpu_tests/"+f, index_col=False, header=None)
    df_X=df.iloc[:,:-1]
    df_Y=df.iloc[:,-1:]

    df_X = df_X.iloc[:,[0,1,2,5]]

    X_train=df_X.iloc[:n_devices*n_samples_device,:]
    y_train=df_Y.iloc[:n_devices*n_samples_device,:]


    X_test=df_X.iloc[n_devices*n_samples_device:,:]
    y_test=df_Y.iloc[n_devices*n_samples_device:,:]

    #scaler = QuantileTransformer(n_quantiles=100)
    #scaler=MinMaxScaler()
    #scaler=StandardScaler()

    X_train=X_train.drop(0, axis=1)
    X_train=X_train.drop(1, axis=1)
    X_train=X_train.drop(2, axis=1)
    #X_train=scaler.fit_transform(X_train)

    X_train=X_train.values.reshape(X_train.shape[0]//window,window)
    train_mean=pd.DataFrame(X_train.mean(axis=0))
    print(train_mean.describe())
    y_train=  [item[0] for item in y_train.values.reshape(y_train.shape[0]//window,window)]
    print(y_train)

    X_test=X_test.drop(0, axis=1)
    X_test=X_test.drop(1, axis=1)
    X_test=X_test.drop(2, axis=1)

    X_test=X_test.values.reshape(X_test.shape[0]//window,window)
    test_mean=pd.DataFrame(X_test.mean(axis=0))
    print(test_mean.describe())
    y_test= [item[0] for item in y_test.values.reshape(y_test.shape[0]//window,window)]
    print(y_test)
    #X_test=scaler.transform(X_test)


    #rf = RandomForestClassifier(n_estimators = 100,random_state=1)
    #rf = XGBClassifier(min_child_weight= 5, max_depth= 20, learning_rate= 0.1, gamma= 0.01, colsample_bytree= 0.5)
    rf = KNeighborsClassifier(n_neighbors=20)
    #rf = GaussianNB()
    rf.fit(X_train, y_train)

    pred=rf.predict(X_test)
    accuracy=rf.score(X_test,y_test)
    print("Accuracy: {}".format(accuracy))
    print(classification_report(y_test, pred, target_names=rf.classes_))

    array=confusion_matrix(y_test, pred)
    df_cm = pd.DataFrame(array)
    plt.figure(figsize = (35,35))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g',xticklabels=rf.classes_, yticklabels=rf.classes_)
    plt.show()

