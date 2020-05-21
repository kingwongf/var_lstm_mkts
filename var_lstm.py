import numpy as np
import pandas as pd
import os
import random
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
register_matplotlib_converters()

from statsmodels.tsa.vector_ar.var_model import VAR

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


mkts = pd.read_excel("interview_dataset_excel_v2.xlsx", index_col='Date',
                     usecols=['Date','US Equity', 'UK Equity', 'Japan Equity', 'Germany Equity',
                              'Canada Equity', 'US Bond', 'UK Bond', 'Japan Bond', 'Germany Bond',
                              'Canada Bond', 'EM Equity']).dropna(axis=0)


def cycle_encode(data, cols):
    for col in cols:
        data[col + '_sin'] = np.sin(2 * np.pi * data[col] / data[col].max())
        data[col + '_cos'] = np.cos(2 * np.pi * data[col] / data[col].max())

    return data


def plot_autocor(name, df):
    plt.figure(figsize=(16, 4))
    timeLags = np.arange(1, 100 * 24)
    plt.plot([df[name].autocorr(dt) for dt in timeLags])
    plt.title(name);
    plt.ylabel('autocorr');
    plt.xlabel('time lags')
    plt.show()



train_date = int(len(mkts)*0.7)
train_ret, test_ret = mkts.iloc[:train_date].pct_change(1).dropna(axis=0), mkts.iloc[train_date:].pct_change(1).dropna(axis=0)

# for mkt_col in mkts.columns:
    # plot_autocor(mkt_col, train_ret)

### FIND BEST VAR ORDER ###
## BEST ORDER 3 BEST AIC: -115.76223844973231
'''

AIC = {}
best_aic, best_order = np.inf, 0

for i in range(1, 253):
    model = VAR(endog=train_ret.values)
    model_result = model.fit(maxlags=i)
    AIC[i] = model_result.aic

    if AIC[i] < best_aic:
        best_aic = AIC[i]
        best_order = i

print('BEST ORDER', best_order, 'BEST AIC:', best_aic)


### PLOT AICs ###

plt.figure(figsize=(14,5))
plt.plot(range(len(AIC)), list(AIC.values()))
plt.plot([best_order-1], [best_aic], marker='o', markersize=8, color="red")
plt.xticks(range(len(AIC)), range(1,253))
plt.xlabel('lags'); plt.ylabel('AIC')
np.set_printoptions(False)
'''
best_order = 3

## fit VAR model with Order 3
var = VAR(endog=train_ret)
var_result = var.fit(maxlags=best_order)
## VAR test prediction
train_pred_var = pd.DataFrame(columns=train_ret.columns)

for i in range(3, len(train_ret)):
    train_pred_var.loc[i - 3] = var_result.forecast(train_ret.iloc[i - 3:i].values, steps=1)[0]  ## rolling

train_pred_var.index = train_ret.iloc[3:].index
## get VAR prediction to feed in neural network
test_pred_var = pd.DataFrame(columns=test_ret.columns)
for i in range(3, len(test_ret)):
    test_pred_var.loc[i - 3] = var_result.forecast(test_ret.iloc[i - 3:i].values, steps=1)[0]  ## rolling

test_pred_var.index = test_ret.iloc[3:].index
test_pred_var = test_pred_var.dropna(axis=0)

## train, test and Xy split set


## not sure should we feed only returns or both returns and prices
## trying just returns for now, feed both later

## trying out 4 approaches for predicting test_y
## 1. using VAR 1 step
## 2. fit raw returns to LSTM to predict 1 step forward y
## 3. fit raw returns and VAR 1 step prediction to LSTM to predict 1 step forward y
## ?? Not sure how to do this yet 4. fit residuals of VAR, (VAR prediction - y_true)_( to predict 1 step forward y

## only raw returns
y_train, X_train = train_ret.shift(1), train_ret.iloc[1:]
y_test, X_test = test_ret.shift(1), test_ret.iloc[1:]

## raw returns with VAR prediction
## back shift as that's the prediction of the current day from yesterday's
train_pred_var, test_pred_var = train_pred_var.add_prefix('var_').shift(-1), test_pred_var.add_prefix('var_').shift(-1)
comb_X_train, comb_X_test = X_train.merge(train_pred_var, left_index=True, right_index=True), X_test.merge(test_pred_var, left_index=True, right_index=True)

comb_X_train.dropna(inplace=True)
comb_X_test.dropna(inplace=True)

y_train = y_train.loc[comb_X_train.index]
y_test = y_test.loc[comb_X_test.index]

X_train = X_train.loc[comb_X_train.index]
X_test = X_test.loc[comb_X_test.index]

## Adding Validation Set ## TODO, to run again for next training
val_split = int(len(X_train)*0.8)
X_train, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
y_train, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

comb_X_train, comb_X_val = comb_X_train.iloc[:val_split], comb_X_train.iloc[val_split:]



## Scaling

scalar_X, scalar_comb_X, scalar_y = StandardScaler(), StandardScaler(), StandardScaler()

scalar_X.fit(X_train), scalar_comb_X.fit(comb_X_train), scalar_y.fit(y_train)
scaled_X_train = scalar_X.transform(X_train)
scaled_X_val = scalar_X.transform(X_val)
scaled_X_test = scalar_X.transform(X_test)

scaled_comb_X_train = scalar_comb_X.transform(comb_X_train)
scaled_comb_X_val = scalar_comb_X.transform(comb_X_val)
scaled_comb_X_test = scalar_comb_X.transform(comb_X_test)

scaled_y_train = scalar_y.transform(y_train)
scaled_y_val = scalar_y.transform(y_val)
scaled_y_test = scalar_y.transform(y_test)

### BUILD DATA GENERATOR ###
seq_length = 25 ## feed 25 days returns to predict 1 forward
generator_train = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=seq_length, batch_size=32)
generator_train_var = TimeseriesGenerator(scaled_comb_X_train, scaled_y_train, length=seq_length, batch_size=32)

generator_val = TimeseriesGenerator(scaled_X_val, scaled_y_val, length=seq_length, batch_size=32)
generator_val_var = TimeseriesGenerator(scaled_comb_X_val, scaled_y_val, length=seq_length, batch_size=32)

generator_test = TimeseriesGenerator(scaled_X_test, scaled_y_test, length=seq_length, batch_size=32)
generator_test_var = TimeseriesGenerator(scaled_comb_X_test, scaled_y_test, length=seq_length, batch_size=32)




def get_model(seq_length, n_feat, n_y):
    opt = RMSprop(lr=0.002)

    inp = Input(shape=(seq_length, n_feat))

    x = LSTM(64)(inp)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(n_y)(x)

    model = Model(inp, out)
    model.compile(optimizer=opt, loss='mse')

    return model

### FIT NEURAL NETWORK WITH 1. RAW DATA , 2. VAR predictions AND RAW DATA ###
tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(),
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)

print('--------', 'train model with VAR fitted values', '--------')

## training two LSTMs,
# raw return only: model,
# raw returns and VAR value: model_var
model_var = get_model(seq_length, len(comb_X_train.columns.tolist()), len(y_train.columns.tolist()))


model_var.fit_generator(generator_train_var, steps_per_epoch= len(generator_train_var),
                        epochs=100, validation_data=generator_val_var, validation_steps = len(generator_val_var),
                        callbacks=[es], verbose = 1)

model = get_model(seq_length, len(X_train.columns.tolist()), len(y_train.columns.tolist()))


model.fit_generator(generator_train, steps_per_epoch= len(generator_train_var),
                        epochs=100, validation_data=generator_val, validation_steps = len(generator_val),
                        callbacks=[es], verbose = 1)

pred_comb_ts_gen = scalar_y.inverse_transform(model_var.predict_generator(generator_test_var))
pred_ts_gen = scalar_y.inverse_transform(model.predict_generator(generator_test))

print(f"no. of things to predict: {y_test.shape}")
print(pred_comb_ts_gen.shape)
print(pred_ts_gen.shape)
print(test_pred_var.shape)


print(len(generator_test_var[1]))

print(scalar_comb_X.inverse_transform(generator_train_var[0]))

################################################################

## Things not working yet
print(len(pred_comb_ts_gen))
print(len(y_test))

print(scaled_comb_X_test.shape)

print(pred_comb_ts_gen)
print(y_test)
print(test_pred_var)
## LSTM expects input to have [samples, timesteps, features]
print(pred_comb_ts_gen.shape)
model_var.predict(scaled_comb_X_test)
pred_test_comb = scalar_y.inverse_transform(model_var.predict(scaled_comb_X_test.reshape(len(scaled_comb_X_test), 25, len(comb_X_train.columns.tolist()))))
pred_test = scalar_y.inverse_transform(model_var.predict(scaled_X_test))

