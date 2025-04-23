import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

df=pd.read_csv("/content/drive/MyDrive/intern/survey lung cancer.csv")
df.head()


df.isna().sum()
df.drop('Unnamed: 16',axis=1)

df.GENDER=df.GENDER.replace({"M":1,"F":0})

df.LUNG_CANCER=df.LUNG_CANCER.replace({"YES":1,"NO":0})

df=df.drop('Unnamed: 16',axis=1)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense


X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
predictions = model.predict(X_test)
threshold = 0.5
absolute_diff = np.abs(predictions.flatten() - y_test.values[:len(predictions)])
accuracy = np.mean(absolute_diff < threshold) * 100
print("Accuracy:", accuracy, "%")