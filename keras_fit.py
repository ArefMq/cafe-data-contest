#!/usr/bin/python

from load_data import load_all, unwrap_result, NORM_F_PRICE, MILLION

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np
import pandas
import time

EPOCH = 5
DATA_OVERWRITE = True
SEED = None


def mean_pred(y_true, y_pred):
    return K.mean(y_pred) * NORM_F_PRICE * 100 * MILLION


def root_mean_squared_error(y_true, y_pred):
    rmse = K.log(K.abs(y_true) + K.epsilon()) - K.log(K.abs(y_pred) + K.epsilon())
    rmse = K.sqrt(K.mean(K.square(rmse), axis=-1))
    return rmse


if SEED is not None:
    np.random.seed(SEED)

[train_in, train_out, prediction_in] = load_all()
input_shape = train_in.shape[1]

# Attempt 2 ---------------------------------------------------------
model = Sequential()
model.add(Dense(20, input_dim=input_shape, kernel_initializer='normal', activation='linear'))
model.add(LeakyReLU(alpha=0.15))
model.add(Dense(1, kernel_initializer='normal', activation='linear'))
model.add(LeakyReLU(alpha=0.15))
model.compile(loss=root_mean_squared_error, optimizer='rmsprop')

# Attempt 1 ---------------------------------------------------------
# model = Sequential()
# model.add(Dense(60, input_dim=input_shape, activation='softmax'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(1))
# # sgd = SGD(lr=0.1)
# # model.compile(loss='kullback_leibler_divergence', optimizer=sgd)
# model.compile(optimizer="rmsprop", loss=root_mean_squared_error,
#               metrics=["accuracy", mean_pred])

model.fit(train_in, train_out, batch_size=128, nb_epoch=EPOCH)
loss_and_metrics = model.evaluate(train_in, train_out, batch_size=128)
result = model.predict_proba(prediction_in)
result = np.floor(unwrap_result(result, prediction_in).astype(float))

print
print 'Loss   ::', loss_and_metrics
print 'Result ::', result

filename = 'result_%s' % (str(time.time()) if not DATA_OVERWRITE else 'last')
np.savetxt('results/%s.txt' % filename, result)
data_frame = pandas.read_csv('results/%s.txt' % filename)
with open('results/%s.csv' % filename, 'w') as f:
    f.write('price\n')
    for d in data_frame.values:
        f.write('%.0f\n' % d[0])
