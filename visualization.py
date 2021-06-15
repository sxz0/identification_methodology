import os
import statistics
import time

from combo.utils.utility import standardizer
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
import umap
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.loci import LOCI
from pyod.models.xgbod import XGBOD
from pyod.models.cof import COF
from pyod.models.loda import LODA
from pyod.models.copod import COPOD
from pyod.models.sod import SOD
from pyod.models.vae import VAE
from pyod.models.lof import LOF, LocalOutlierFactor
from pyod.models.lscp import LSCP
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.iforest import IsolationForest, IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.knn import KNN
from pyod.models.combination import moa, aom, median, majority_vote
from pyod.models.abod import ABOD
import warnings
import seaborn as sb
from scipy.stats import gaussian_kde

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")
n_samples_device = 200#400  # GPU int(10000) #CPU
datasets = os.listdir("datatests")

f = "feat/stability_test_4min_2.csv"

colors = {'1': "b", '2': "g", '3': "c", '4': "r", '5': "m"}

df = pd.read_csv("gpu_tests/" + f, index_col=False, header=None)
df.replace(np.inf, np.nan, inplace=True)
df.fillna(0, inplace=True)
#df = df.iloc[4000:, :]
df_X = df.iloc[:, :-1]
df_Y = df.iloc[:, -1:]
print(df_X.shape)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_X.describe())

for i in range(5, 20):
    f_hist = []

    feat_to_plot = i

    for n in range(0, df.shape[0] // n_samples_device):
        df_X_selec = df_X[n * n_samples_device:n * n_samples_device + n_samples_device]
        df_Y_selec = df_Y[n * n_samples_device:n * n_samples_device + n_samples_device]
        df_X_selec = df_X_selec[df_X_selec[2] != 0]
        df_X_selec = df_X_selec.loc[(df_X_selec[feat_to_plot]>=4084) & (df_X_selec[feat_to_plot]<=4092)]
        l = df_X_selec[feat_to_plot]
        l=(l-4084)/8
        print(l)
        print(df_Y_selec.iloc[0,0]+" "+str(l.mean()))
        f_hist.append(l)

    min = 0# df_X[feat_to_plot].min()
    max = 1#df_X[feat_to_plot].max()
    print(len(f_hist))
    n = 0
    plt.figure(figsize=(5, 5))
    for f in f_hist:
        density = gaussian_kde(f)
        xs = np.linspace(min, max, 1000)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        c = colors[str(n % 5 + 1)]
        # plt.plot(xs,density(xs),label=str(n%5+1),color=c)
        # plt.hist(f, bins = 100,label=str(n%5+1),color=c)
        sb.kdeplot(f, bw=0.5, fill=False, color=c)#, label=str(n % 5 + 1)
        n = n + 1

    plt.axvline(x=(4088.36-4084)/8,color="b", linestyle='dashed',label="Device 1")
    plt.axvline(x=(4087.4288-4084)/8,color="g", linestyle='dashed',label="Device 2")
    plt.axvline(x=(4087.89-4084)/8,color="c", linestyle='dashed',label="Device 3")
    plt.axvline(x=(4087.13-4084)/8,color="r", linestyle='dashed',label="Device 4")
    plt.axvline(x=(4087.70-4084)/8,color="m", linestyle='dashed',label="Device 5")
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    plt.legend(prop={'size': 15})
    plt.show()
