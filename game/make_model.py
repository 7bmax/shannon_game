import tensorflow as tf
import pandas as pd

from tensorflow.keras import models
from keras.layers import Dense

import time

def train_and_save_new_model():
    train_df = pd.read_csv("/home/maxime/Documents/shannon_game/data/train_dataset.csv")
    X_train = train_df.loc[:, train_df.columns != 'action']
    y_train = train_df.loc[:, train_df.columns == 'action']

    depth = train_df.shape[1] - 1

    model = models.Sequential()
    # model.add(tf.keras.layers.LSTM(depth, 
    #                               input_shape=(depth, len(X_train)), 
    #                               activation='tanh')) 
    # model.add(Dense(depth, activation='relu'))   
    # model.add(Dense(depth*2, activation='relu')) 
    # model.add(Dense(1, activation='sigmoid')) 
    model.add(Dense(len(X_train), 
                    input_dim=depth, 
                    activation='tanh'))
    model.add(Dense(depth, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1000)

    model.save('/home/maxime/Documents/shannon_game/models/tensor_model')
    pass

if __name__ == '__main__':
    a = True
    while(a):
        train_and_save_new_model()
        time.sleep(1)