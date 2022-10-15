# importing the required libraries
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# reading the data
df = pd.read_excel('F:/合作博弈数据.xlsx')

# imputing missing values in Item_Weight by median and Outlet_Size with mode
df['Item_Weight'].fillna(df['Item_Weight'].median(), inplace=True)
df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)

# creating a broad category of type of Items
df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda df: df[0:2])
df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})

df['Item_Type_Combined'].value_counts()

# operating years of the store
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

# modifying categories of Item_Fat_Content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
df['Item_Fat_Content'].value_counts()

# label encoding the ordinal variables
le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type', 'Outlet']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

# one hot encoding the remaining categorical variables
df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                 'Item_Type_Combined', 'Outlet'])

# dropping the ID variables and variables that have been used to extract new variables
df.drop(['Item_Type', 'Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)

# separating the dependent and independent variables
X = df.drop('Item_Outlet_Sales', 1)
y = df['Item_Outlet_Sales']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Need to load JS vis in the notebook
shap.initjs()

xgb_model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001, random_state=0)
xgb_model.fit(X_train, y_train)

y_predict = xgb_model.predict(X_test)
mean_squared_error(y_test, y_predict) ** (0.5)

# 需要计算出每个夏普利值
# shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns)
