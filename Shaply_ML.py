# importing the required libraries
import pandas as pd
import numpy as np
import shap
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree

import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
plt.figure(dpi=300, figsize=(24, 8))

import warnings

warnings.filterwarnings('ignore')

shap.initjs()

# reading the data
df = pd.read_csv('F:/shapely_value.csv')
df_list = df.values.tolist()
shap_values = np.array(df_list)
index = ['Shanghai', 'Guangdong', 'Hainan', 'Guangxi', 'Tianjin', 'Hebei', 'Liaoning', 'Jiangsu', 'Zhejiang', 'Fujian',
         'Shandong']

df_1 = pd.read_excel(r"F:/合作博弈数据.xlsx")
feature = df_1["协调耦合(X)"].values.tolist()

'''
蜂群图中 shap_values 已知, feature_value是什么?
'''
shap.summary_plot(shap_values, features=feature, feature_names=index)
# shap.plots.beeswarm(shap_values=shap_values, feature_names=index)
