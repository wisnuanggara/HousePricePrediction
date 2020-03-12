#importing the package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.regression.linear_model as sm

#importing the data
dataset = pd.read_csv('data.csv')
dataset= dataset.drop(['date','country','street','statezip'],axis=1).values



#encoding categorical data
labelEncoder_X = LabelEncoder()
dataset[:,13]= labelEncoder_X.fit_transform(dataset[:,13])
onehotencoder = OneHotEncoder(categorical_features=[13])
dataset = onehotencoder.fit_transform(dataset).toarray()

#avoiding the dummy variable trap
dataset=dataset[:, 1:]

#select best variable with high correlation
dataset.corr()[43].sort_values()


data.columns

# how many data we have
data.shape

#checking price range
data.price.describe()

#to view better number, not with e-
pd.set_option('display.float_format', lambda x: '%.2f' % x)


data[data.price == 26590000]


data.sort_values('price', ascending=False).head(10)

data.corr()['price'].sort_values()

data[['sqft_living', 'sqft_above', 'bathrooms', 'view', 'sqft_basement', 'price']].head()

print(data.columns)
print(len(data.columns))

data[['price']].describe()

#check what house that don't have price (price=0)
data[data['price'] == 0]

#check how many data
data[data['price'] == 0].shape[0]

#check what house that don't have price (price=0)
data[data['price'] == 26590000]

#what should you do?
#in this case I will do something, drop value that not on range
iqr = data['price'].describe()['75%'] - data['price'].describe()['25%']
lower_bound = data['price'].describe()['25%'] - (1.5*iqr)
upper_bound = data['price'].describe()['75%'] + (1.5*iqr)
print("IQR equals {}".format(iqr))
print("Lower bound of price is {}".format(lower_bound))
print("Upper bound of price is {}".format(upper_bound))

#just go on with data itself
data_clean = data.copy()
data_clean = data[(data.price > 0) & (data.price <= upper_bound)]
data_clean.shape

data_clean[['price']].describe()


+ labelEnocder_data = LabelEncoder()
data_clean[:, 13] = labelEnocder_data.fit_transform(data_clean[:,13])


data_clean.to_csv("Data Clean.csv",index=False)