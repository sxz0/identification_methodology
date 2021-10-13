import os
import statistics
import time
from imblearn.over_sampling import *
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
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
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
#import umap

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

#General parameters for the experiments
dataset_dir="../datasets"
n_samples_device = 800 #10000
window=10
n_recursive_windows=10
jump=10
initial_window=window
test_size=0.40
feat_list=[1,4,5,6]
n_feat_selec=15
model="3_"
model_2="4_"
#Dataset to be read and processed
dataset_name="sleep_2min_800_longhash_noreboot_5fings.csv"#"sleep_2min_400.csv"
mac_model_file="../MAC-Model.txt"

df=pd.read_csv(dataset_dir+"/"+dataset_name, index_col=False, header=None)

#df=df.iloc[:df.shape[0]//2,:] #Half of the df

final_df = pd.DataFrame()
df_X = df.iloc[:, :-1]
df_Y = df.iloc[:, -1:]
print(df.describe())

df_X = df_X.iloc[:, feat_list]  # 0,1,2
df_Y.columns=[0]

#### Add model to the label to add clarity in the plots #####
mac_model={}
with open(mac_model_file) as f:
    for line in f:
        p=line.split(" ")
        mac_model[p[0]]=p[3]
df_Y[0]=df_Y[0].apply(lambda x: mac_model[str(x)]+"_"+str(x))


for n in range(0,df.shape[0]//n_samples_device):
    df_X_selec = df_X[n * n_samples_device:n * n_samples_device + n_samples_device]
    df_Y_selec = df_Y[n * n_samples_device:n * n_samples_device + n_samples_device]

    window=initial_window

    temp_df = pd.DataFrame()
    temp_df = temp_df.append(df_Y_selec)
    for c in range(n_recursive_windows):#(df_X_selec.shape[1]): #(1):
        for f in range(df_X_selec.shape[1]):
            df_X_selec_c=df_X_selec.iloc[:, f]
            temp_df["mean_"+str(window)+"_"+str(f)] = df_X_selec_c.rolling(window).mean()
            temp_df["min_"+str(window)+"_"+str(f)]=df_X_selec_c.rolling(window).min()
            temp_df["max_"+str(window)+"_"+str(f)]=df_X_selec_c.rolling(window).max()
            temp_df["median_"+str(window)+"_"+str(f)] = df_X_selec_c.rolling(window).median()
            #temp_df["stdev_"+str(window)+"_"+str(f)] = df_X_selec_c.rolling(window).std()
            #temp_df["skew_"+str(window)] = df_X_selec_c.rolling(window).skew()
            #temp_df["kurt_"+str(window)] = df_X_selec_c.rolling(window).kurt()
            temp_df["sum_"+str(window)+"_"+str(f)] = df_X_selec_c.rolling(window).sum()
        window+=jump

    temp_df["Y"] = df_Y_selec

    temp_df = temp_df.iloc[:, 1:]

    final_df = final_df.append(temp_df)

print(final_df.shape)
final_df.dropna(inplace=True)

print(final_df)
print(final_df.shape)

final_df=final_df[final_df["Y"].str.contains(model)+final_df["Y"].str.contains(model_2)]

df_X=final_df.iloc[:,:-1]
df_Y=final_df.iloc[:,-1:]
X_train,X_test, y_train,y_test = train_test_split(df_X,df_Y, test_size=test_size,shuffle=False)#, stratify=df_Y) #IMPORTANT

#X_train,X_test, y_train,y_test = train_test_split(X_test,y_test, test_size=0.5,shuffle=False)#, stratify=df_Y) #IMPORTANT


####DATA AUGMENTATION
"""
dummy_X=pd.DataFrame(columns=df_X.columns)
dummy_Y=pd.DataFrame(columns=df_Y.columns)
print(X_train.shape)
for i in range(0,1000):
    data=[{'mean_0':0.0, 'min_0':0.0, 'max_0':0.0, 'median_0':0.0, 'sum_0':0.0}]
    dummy_X=dummy_X.append(data)
    dummy_Y=dummy_Y.append([{"Y":"dummy_Y"}])
X_train=X_train.append(dummy_X)
y_train=y_train.append(dummy_Y)
print(X_train.shape)
X_train.dropna(inplace=True,axis=1)
oversample=SVMSMOTE()
X_train,y_train=oversample.fit_resample(X_train,y_train)
X_train=X_train[X_train.median_0 != 0]
y_train=y_train[y_train.Y != "dummy_Y"]
print(X_train.shape)
"""

rf = RandomForestClassifier(n_estimators = 100,bootstrap=False,n_jobs=6,random_state=42)
#rf=ExtraTreesClassifier(n_estimators=500)
#rf = DecisionTreeClassifier()
#rf = XGBClassifier()#max_depth= 20, learning_rate= 0.1, gamma= 0.01, colsample_bytree= 0.5)
#rf = KNeighborsClassifier(n_neighbors=15)
#rf = GaussianNB()
#rf= svm.SVC(kernel='rbf')

#rf=VotingClassifier([('rf',RandomForestClassifier(n_estimators = 500)), ('et',ExtraTreesClassifier(n_estimators=500)),
                     #('xg',XGBClassifier())],n_jobs=6,voting="hard")#max_depth= 20, learning_rate= 0.1, gamma= 0.01, colsample_bytree= 0.5))])
                     #,('knn',KNeighborsClassifier(n_neighbors=8)), ('svm',svm.SVC(kernel='rbf'))])

rf.fit(X_train, y_train)

pred=rf.predict(X_test)
accuracy=rf.score(X_test,y_test)
print("Accuracy: {}".format(accuracy))
print(classification_report(y_test, pred, target_names=rf.classes_))

array=confusion_matrix(y_test, pred)
array=array.astype('float') / array.sum(axis=1)[:, np.newaxis]
array=array.round(2)
df_cm = pd.DataFrame(array)
plt.figure(figsize = (35,35))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g',xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.show()

feat_labels=X_train.columns
importances=rf.feature_importances_
indices=np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))


X_train=X_train.iloc[:,indices[:n_feat_selec]]
X_test=X_test.iloc[:,indices[:n_feat_selec]]

rf.fit(X_train, y_train)

pred=rf.predict(X_test)
accuracy=rf.score(X_test,y_test)
print("Accuracy: {}".format(accuracy))
print(classification_report(y_test, pred, target_names=rf.classes_))

array=confusion_matrix(y_test, pred)
array=array.astype('float') / array.sum(axis=1)[:, np.newaxis]
array=array.round(2)
df_cm = pd.DataFrame(array)
plt.figure(figsize = (35,35))
plt.yticks(rotation=45)
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g',xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.show()