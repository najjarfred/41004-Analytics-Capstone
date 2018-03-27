'''
Models built using sklearn. 
Commented out functions uses Keras
'''

import os
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Globals to control location of data
home_dir = "/Users/felixgu/Desktop/Analytics Capstone/"  
data_file = home_dir + "house_valuation.zip"

def load_test():

    # Use a composite key, address and streed if for unique identification
    index_col = ['AddressID', 'StreetID', 'SuburbID']

    # Use these columns for training
    use_cols = [ 'AddressID', 'StreetID', 'SuburbID',
       'PostcodeID', 'StateID', 'Postcode', 'EventPrice',
       'FirstAdvertisedEventPrice', 'LastAdvertisedEventPrice', 'PropertyType', 
       'AreaSize', 'Bedrooms', 'Baths', 'Parking',
       'HasStudy', 'HasSeparateDining', 'HasFamilyRoom', 'HasSunroom',
       'HasBilliardRoom', 'HasRumpusRoom', 'HasFireplace', 'HasWalkInWardrobe',
       'HasCourtyard', 'HasInternalLaundry', 'HasHeating', 'HasSauna',
       'HasAirConditioning', 'HasBalcony', 'HasBarbeque',
       'HasPolishedTimberFloor', 'HasEnsuite', 'HasSpa', 'HasGarage',
       'HasLockUpGarage', 'HasPool', 'HasTennisCourt', 'HasBeenRenovated',
       'HasAlarm', 'HasWaterView', 'HasHarbourView', 'HasOceanView',
       'HasCityView', 'HasBushView', 'HasDistrictView', 'HasBayView',
       'HasParkView', 'HasRiverView', 'HasMountainView', 'Latitude',
       'Longitude' ]

    # Define dtypes
    dtypes = {
        'AddressID': 'str', 
        'StreetID': 'str', 
        'SuburbID': 'str',
        'HasInternalLaundry': 'str',
        'HasPolishedTimberFloor': 'str',
        'HasHeating': 'str',
        'HasPolished': 'str',
        'HasBarbeque': 'str',
        'HasCityView': 'str',
        'HasBushView': 'str',
        'HasDistrictView': 'str',
        'HasBayView': 'str',
        'HasParkView': 'str',
        'HasRiverView': 'str',
        'HasMountainView': 'str',
        'Latitude': 'float',
        'Longitude': 'float'
    }

    # Date Parsing
    # dates = ['EventDate', 'FirstAdvertisedEventDate', 'LastAdvertisedEventDate', 'ContractDate', 'TransactionDate']

    # Read from file
    data = pd.read_csv(data_file, dtype=dtypes, index_col=index_col, usecols=use_cols)

    # Replace True / False with 1 / 0
    data = data.applymap(lambda x: 1 if x == 'True' or x == True or x == 'TRUE' else x)
    data.fillna(0, inplace=True)

    # Convert PropertyType column to numbers
    data['PropertyType'].replace('Unit', 0, inplace=True)
    data['PropertyType'].replace('House', 1, inplace=True)
    data['PropertyType'].replace('Duplex', 2, inplace=True)
    data['PropertyType'].replace('Townhouse', 3, inplace=True)
    data['PropertyType'].replace('Semi', 4, inplace=True)
    data['PropertyType'].replace('Cottage', 4, inplace=True)
    data['PropertyType'].replace('Villa', 5, inplace=True)
    data['PropertyType'].replace('Terrace', 6, inplace=True)
    data['PropertyType'].replace('Studio', 7, inplace=True)
    data['PropertyType'] = data['PropertyType'].astype(int)

    return data

# Load data using function
data = load_test()

# Get target class
prices = data['EventPrice'].as_matrix()

# Drop the target class in training data
data.drop('EventPrice', axis=1, inplace=True)
# Convert to numpy array for the train-test splitter
data = np.asarray(data)

x_train, x_test, y_train, y_test = train_test_split(data, prices, test_size=0.3, random_state=1)

# Confirm Dimensions of data
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Build basic sklearn Linear regression model
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train, y_train)
print("LinearRegression")

# Predict
y_pred_tr = clf.predict(x_train)
y_pred_te = clf.predict(x_test)

# Predictions of first 10 in set
print(y_test[:10])
print(y_pred_te[:10])

# The coefficients
print('Coefficients: \n', clf.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred_te))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred_te))