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
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import layers

from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class var_lstm:
    def __init__(self, ret_freq):
        mkts = pd.read_excel("interview_dataset_excel_v2.xlsx", index_col='Date',
                             usecols=['Date','US Equity', 'UK Equity', 'Japan Equity', 'Germany Equity',
                                      'Canada Equity', 'US Bond', 'UK Bond', 'Japan Bond', 'Germany Bond',
                                      'Canada Bond', 'EM Equity']).dropna(axis=0)
        self.mkts = mkts
        train_date = int(len(mkts)*0.7)
        self.best_order = 3
        self.train_ret, self.test_ret = mkts.iloc[:train_date].pct_change(ret_freq).dropna(axis=0), mkts.iloc[train_date:].pct_change(ret_freq).dropna(axis=0)
        self.train_pred_var = pd.DataFrame(columns=self.train_ret.columns)
        self.test_pred_var = pd.DataFrame(columns=self.test_ret.columns)

        self.seq_length = 25  ## feed 25 days returns to predict ret_freq forward

    def find_opt_lag(self, plot=False):

        AIC = {}
        best_aic, best_order = np.inf, 0

        for i in range(1, 253):
            model = VAR(endog=self.train_ret.values)
            model_result = model.fit(maxlags=i)
            AIC[i] = model_result.aic

            if AIC[i] < best_aic:
                best_aic = AIC[i]
                best_order = i

        print('BEST ORDER', best_order, 'BEST AIC:', best_aic)

        if plot:
            plt.figure(figsize=(14, 5))
            plt.plot(range(len(AIC)), list(AIC.values()))
            plt.plot([best_order - 1], [best_aic], marker='o', markersize=8, color="red")
            plt.xticks(range(len(AIC)), range(1, 253))
            plt.xlabel('lags')
            plt.ylabel('AIC')
        return best_order

    def _train_VAR(self):
        ## fit VAR model with Order 3
        var = VAR(endog=self.train_ret)
        var_result = var.fit(maxlags=self.best_order)
        ## VAR test prediction

        for i in range(self.best_order, len(self.train_ret) + 1):
            self.train_pred_var.loc[i - self.best_order] = var_result.forecast(self.train_ret.iloc[i - self.best_order:i].values, steps=1)[
                0]  ## rolling

        for i in range(self.best_order, len(self.test_ret) + 1):
            self.test_pred_var.loc[i - self.best_order] = var_result.forecast(self.test_ret.iloc[i - self.best_order:i].values, steps=1)[
                0]  ## rolling


    def _prep_VAR_LSTM_X(self):
        train_var_lstm_X = self.train_pred_var.copy().add_prefix('var_')
        train_var_lstm_X.index = self.train_ret.iloc[self.best_order - 1:].index
        train_var_lstm_X = train_var_lstm_X.merge(self.train_ret, how='left', left_index=True, right_index=True)

        test_var_lstm_X = self.test_pred_var.copy().add_prefix('var_')
        test_var_lstm_X.index = self.test_ret.iloc[self.best_order - 1:].index
        test_var_lstm_X = test_var_lstm_X.merge(self.test_ret, how='left', left_index=True, right_index=True)



        ## need to drop the last row as it's a prediction for the non-exisisting date
        self.test_pred_var = self.test_pred_var.iloc[:-1]
        ## VAR uses obs until the best obs, so prediction should start from there
        self.test_pred_var.index = self.test_ret.iloc[self.best_order:].index
        self.train_var_lstm_X = train_var_lstm_X
        self.test_var_lstm_X = test_var_lstm_X

    def _reset_train_test_index(self):
        self.train_ret = self.train_ret.loc[self.train_var_lstm_X.index]
        self.test_ret = self.test_ret.loc[self.test_var_lstm_X.index]

    def _add_val_set(self):

        ## Adding Validation Set ##
        val_split = int(len(self.train_ret) * 0.8)
        self.val_ret = self.train_ret[val_split:]
        self.train_ret = self.train_ret[:val_split]
        ## Adding Validation Set for VAR Combined Input ##
        self.val_var_lstm_X = self.train_var_lstm_X[val_split:]
        self.train_var_lstm_X = self.train_var_lstm_X[:val_split]


    def _scaling(self):
        ## Scaling
        '''

        :return: two scalars for later inverse transform
        '''

        scalar, scalar_var_lstm_X = StandardScaler(), StandardScaler()

        scalar.fit(self.train_ret)
        scalar_var_lstm_X.fit(self.train_var_lstm_X)

        self.scaled_train_ret = scalar.transform(self.train_ret)
        self.scaled_val_ret = scalar.transform(self.val_ret)
        self.scaled_test_ret = scalar.transform(self.test_ret)

        self.scaled_train_var_lstm_X = scalar_var_lstm_X.transform(self.train_var_lstm_X)
        self.scaled_val_var_lstm_X = scalar_var_lstm_X.transform(self.val_var_lstm_X)
        self.scaled_test_var_lstm_X = scalar_var_lstm_X.transform(self.test_var_lstm_X)

        self.scalar = scalar



    def _ts_generator(self):
        ### time series generator
        seq_length = self.seq_length
        self.generator_train = TimeseriesGenerator(self.scaled_train_ret, self.scaled_train_ret, length=seq_length, batch_size=32)
        self.generator_train_var_lstm = TimeseriesGenerator(self.scaled_train_var_lstm_X, self.scaled_train_ret, length=seq_length,
                                                       batch_size=32)

        self.generator_val = TimeseriesGenerator(self.scaled_val_ret, self.scaled_val_ret, length=seq_length, batch_size=32)
        self.generator_val_var_lstm = TimeseriesGenerator(self.scaled_val_var_lstm_X, self.scaled_val_ret, length=seq_length,
                                                     batch_size=32)

        self.generator_test = TimeseriesGenerator(self.scaled_test_ret, self.scaled_test_ret, length=seq_length, batch_size=32)
        self.generator_test_var_lstm = TimeseriesGenerator(self.scaled_test_var_lstm_X, self.scaled_test_ret, length=seq_length,
                                                      batch_size=32)


    def gen_model(self, seq_length, n_feat, n_y):
        model = tf.keras.Sequential()
        model.add(layers.LSTM(128, return_sequences=True, input_shape=(seq_length, n_feat)))
        model.add(layers.LSTM(128))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_y))
        model.compile(loss='mse', optimizer='RMSprop')
        return model

    def fit_models(self):

        generator_train = self.generator_train
        generator_val = self.generator_val
        generator_train_var_lstm = self.generator_train_var_lstm
        generator_val_var_lstm = self.generator_val_var_lstm

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


        ## training two LSTMs,
        # raw return only: model,
        # raw returns and VAR value: model_var

        ## params order: seq_length, n_feat, n targets
        n_feat_lstm = len(self.test_ret.columns)
        n_feat_var_lstm = len(self.test_var_lstm_X.columns)
        n_targets = len(self.test_ret.columns)


        model_lstm = self.gen_model(self.seq_length, n_feat_lstm, n_targets)


        model_lstm.fit_generator(generator_train, steps_per_epoch= len(generator_train),
                                epochs=100, validation_data=generator_val, validation_steps = len(generator_val),
                                callbacks=[es], verbose = 1)


        model_var_lstm = self.gen_model(self.seq_length, n_feat_var_lstm, n_targets)


        model_var_lstm.fit_generator(generator_train_var_lstm, steps_per_epoch= len(generator_train_var_lstm),
                                epochs=100, validation_data=generator_val_var_lstm, validation_steps = len(generator_val_var_lstm),
                                callbacks=[es], verbose = 1)
        return model_lstm, model_var_lstm

    def predict(self, generator_test, model):
        return self.scalar.inverse_transform(model.predict_generator(generator_test))

    def preprocessing(self):
        self._train_VAR()
        self._prep_VAR_LSTM_X()
        self._reset_train_test_index()
        self._add_val_set()
        self._scaling()
        self._ts_generator()

    def metrics_plot(self,func_metrics, metric_name, dfs):
        compare_true_ret, compare_var_ret, pred_LSTM_ret, pred_VAR_LSTM_ret = dfs

        metric_df = pd.DataFrame()
        for col in compare_true_ret.columns:
            col_metric = pd.Series({'VAR': func_metrics(compare_true_ret[col], compare_var_ret[col]),
                                    'LSTM': func_metrics(compare_true_ret[col], pred_LSTM_ret[col]),
                                    'VAR_LSTM': func_metrics(compare_true_ret[col], pred_VAR_LSTM_ret[col])},
                                   name=col)
            metric_df = metric_df.append(col_metric)
        metric_df.plot.bar()
        plt.ylabel(metric_name)
        plt.title('price returns')
        plt.show()
        plt.close()

    def ret2price(self,ret_df):
        return self.mkts.loc[ret_df.index].shift(1) * (1 + ret_df)

    def performance_analysis(self, compare_true_ret=None, compare_var_ret=None, pred_LSTM_ret=None, pred_VAR_LSTM_ret=None):
        if all(x is None for x in [compare_true_ret, compare_var_ret, pred_LSTM_ret, pred_VAR_LSTM_ret]):
            compare_true_ret = pd.read_pickle("compare_true_ret.pkl")
            compare_var_ret = pd.read_pickle("compare_var_ret.pkl")
            pred_LSTM_ret = pd.read_pickle("pred_LSTM_ret.pkl")
            pred_VAR_LSTM_ret = pd.read_pickle("pred_VAR_LSTM_ret.pkl")

        dfs = [compare_true_ret, compare_var_ret, pred_LSTM_ret, pred_VAR_LSTM_ret]
        self.metrics_plot(r2_score, 'R2', dfs)
        self.metrics_plot(mean_absolute_error, 'MAE', dfs)


        compare_true_price = self.ret2price(compare_true_ret)
        compare_var_price = self.ret2price(compare_var_ret)
        pred_LSTM_price = self.ret2price(pred_LSTM_ret)
        pred_VAR_LSTM_price = self.ret2price(pred_VAR_LSTM_ret)

        for col in compare_true_ret.columns:
            compare_true_price[col].plot(label='True')
            compare_var_price[col].plot(label='VAR')
            pred_LSTM_price[col].plot(label='LSTM')
            pred_VAR_LSTM_price[col].plot(label='VAR_LSTM')
            plt.ylabel("Price")
            plt.legend()
            plt.show()

    def main(self, write=True):
        self.preprocessing()
        model_lstm, model_var_lstm = self.fit_models()
        pred_lstm_arr = self.predict(self.generator_test, model_lstm)
        pred_var_lstm_arr = self.predict(self.generator_test_var_lstm, model_var_lstm)

        ## tidying up index
        compare_true_ret = self.test_ret.iloc[self.seq_length:]

        compare_var_ret = self.test_pred_var.loc[compare_true_ret.index]

        pred_LSTM = pd.DataFrame(pred_lstm_arr, index=compare_true_ret.index, columns=compare_true_ret.columns)
        pred_VAR_LSTM = pd.DataFrame(pred_var_lstm_arr, index=compare_true_ret.index, columns=compare_true_ret.columns)

        ## save as pickles for plotting
        if write:
            for name_df, df in zip(["compare_true_ret", "compare_var_ret", "pred_LSTM_ret", "pred_VAR_LSTM_ret"],
                                   [compare_true_ret, compare_var_ret, pred_LSTM, pred_VAR_LSTM]):
                df.to_pickle(f"{name_df}.pkl")

            model_lstm.save("lstm")
            model_var_lstm.save("var_lstm")

        self.performance_analysis(compare_true_ret, compare_var_ret, pred_LSTM, pred_VAR_LSTM)





print(var_lstm(ret_freq=20).find_opt_lag(plot=True))
# m = var_lstm(ret_freq=20).main(write=False)

exit()