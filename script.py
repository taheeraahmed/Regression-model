import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer

df = pd.read_csv('admissions_data.csv')
df = df.drop(['Serial No.'], axis=1)

#print(df.describe())

# Splitting up the data into two parts:
## Feature being all columns but the last and
## Labels being the last column
features = df.iloc[:,0:-1]
labels = df.iloc[:,-1]

# Splitting the data into train and test
features_training_set, features_test_set, labels_training_set, labels_test_set = train_test_split(features,labels, test_size=0.33, random_state=42)

# Standardizing the data
## Need to standardize data which is 
ct = ColumnTransformer([('normalize', Normalizer(), ['GRE Score', 'TOEFL Score', 'CGPA'])], remainder='passthrough')

features_train_stand = ct.fit_transform(features_training_set)
features_test_stand = ct.transform(features_test_set)

# Creating the neural network model

def create_model(features):
    model = Sequential(name="NeuralNet")

    # Adding the input layer to model
    # num_features is the number of columns in df
    num_features = features.shape[1] 
    input = layers.InputLayer(input_shape=num_features,)
    model.add(input) 

    # Adding the hidden layer to the model 
    # using a RELU function for the activation 
    model.add(Dense(20, activation='relu'))

    # Adding the output layer to model 
    model.add(Dense(1))

    # Adding an optimizer to the model 
    # using mean-squared error (MSE) as the loss function, because it's a regression model
    # and mean-average error (MAE) as the metric
    # choosing Adam as optimizer with a learning rate = 0.01
    opt = Adam(learning_rate=0.1)
    model.compile(loss ='mse', metrics =['mae'], optimizer = opt)


    print(model.summary())
    return model

# Creating the model by passing in the features of the training set
model = create_model(features_training_set)

# Training and fitting the model
model.fit(features_training_set, labels_training_set, epochs = 100, batch_size = 6, verbose = 1)

# Evaluating the model and printing stuff
val_mse, val_mae = model.evaluate(features_training_set, labels_training_set, verbose = 0)

print("\nACCRUACY")
print(f"The value of mean-average error is: {val_mae}")
print(f"The value of mean-squared error is: {val_mse}")


