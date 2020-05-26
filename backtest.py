import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt
from tabulate import tabulate

class backtesting:
    def __init__(self,ret_freq, long_short):
        self.ret_freq = ret_freq
        self.compare_true_ret = pd.read_pickle(f"compare_true_ret_{self.ret_freq}.pkl")

        ## need to shift backward as it's a 1 step forward prediction
        self.compare_var_ret = pd.read_pickle(f"compare_var_ret_{self.ret_freq}.pkl").shift(-1)
        self.pred_LSTM_ret = pd.read_pickle(f"pred_LSTM_ret_{self.ret_freq}.pkl").shift(-1)
        self.pred_LSTM_X_ret = pd.read_pickle(f"pred_LSTM_X_ret_{self.ret_freq}.pkl").shift(-1)
        self.pred_VAR_LSTM_ret = pd.read_pickle(f"pred_VAR_LSTM_ret_{self.ret_freq}.pkl").shift(-1)

        train_ret = pd.read_pickle(f"train_ret_{self.ret_freq}.pkl")
        val_ret = pd.read_pickle(f"val_ret_{self.ret_freq}.pkl")
        train_ret = train_ret.append(val_ret).sort_index()

        df = pd.read_excel("dataset.xlsx", index_col='Date').dropna(axis=0)
        self.mkts = df[['US Equity', 'UK Equity', 'Japan Equity', 'Germany Equity',
                   'Canada Equity', 'US Bond', 'UK Bond', 'Japan Bond', 'Germany Bond',
                   'Canada Bond', 'EM Equity']]
        self.mkts_ret = self.mkts.pct_change(1)
        # self.rebalancing_dates = self.compare_true_ret.resample('1m').first().index

        w_bounds = (-1, 1) if long_short else (0, 1)
        vcv = train_ret.cov()
        mu = train_ret.mean()
        eff = EfficientFrontier(expected_returns=mu, cov_matrix=vcv, weight_bounds=w_bounds)
        self.init_weights = eff.max_sharpe(risk_free_rate=0)
        self.train_ret = train_ret
        self.w_bounds = w_bounds

    def new_w(self, new_pred, date):
        new_ret = self.train_ret.append(new_pred)
        new_vcv = new_ret.cov()
        new_mu = new_ret.mean()
        eff = EfficientFrontier(expected_returns=new_mu, cov_matrix=new_vcv, weight_bounds=self.w_bounds)
        return pd.DataFrame(eff.max_sharpe(risk_free_rate=0), index= [date])

    def new_port_ret(self, curr_pred, rebalancing_date):
        new_w = self.new_w(curr_pred.loc[:rebalancing_date], rebalancing_date)
        return (new_w*self.compare_true_ret.shift(-1).loc[rebalancing_date]).sum(axis=1)

    def daily_port_ret(self, w_df):
        ret_df = pd.DataFrame(index=self.mkts_ret.index).merge(w_df, how='left', left_index=True,
                                                      right_index=True).ffill().dropna(
            axis=0) * self.mkts_ret.shift(-1)
        ret_df = ret_df.dropna(axis=0).sum(axis=1)
        return ret_df
    def main(self):

        # print(self.train_ret)

        # print(self.compare_var_ret)

        ret_df = pd.DataFrame()

        w_MVO_df = pd.DataFrame()
        w_var_df = pd.DataFrame()
        w_LSTM_df = pd.DataFrame()
        w_LSTM_X_df = pd.DataFrame()
        w_VAR_LSTM_df = pd.DataFrame()
        for rebalancing_date in self.compare_true_ret.index:
            # print(f"{rebalancing_date}: {self.compare_true_ret.loc[:rebalancing_date]})")
            # print(f"{rebalancing_date}: {self.compare_var_ret.loc[:rebalancing_date]})")

            w_MVO_df = w_MVO_df.append(self.new_w(self.compare_true_ret.shift(1).loc[:rebalancing_date], rebalancing_date))
            w_var_df = w_var_df.append(self.new_w(self.compare_var_ret.loc[:rebalancing_date], rebalancing_date))
            w_LSTM_df =  w_LSTM_df.append(self.new_w(self.pred_LSTM_ret.loc[:rebalancing_date], rebalancing_date))
            w_LSTM_X_df =  w_LSTM_X_df.append(self.new_w(self.pred_LSTM_X_ret.loc[:rebalancing_date], rebalancing_date))
            w_VAR_LSTM_df =  w_VAR_LSTM_df.append(self.new_w(self.pred_VAR_LSTM_ret.loc[:rebalancing_date], rebalancing_date))



            port_MVO_ret = self.new_port_ret(self.compare_true_ret.shift(1), rebalancing_date).values[0]
            port_var_ret = self.new_port_ret(self.compare_var_ret, rebalancing_date).values[0]
            port_LSTM_ret = self.new_port_ret(self.pred_LSTM_ret, rebalancing_date).values[0]
            port_LSTM_X_ret = self.new_port_ret(self.pred_LSTM_X_ret, rebalancing_date).values[0]
            port_VAR_LSTM_ret = self.new_port_ret(self.pred_VAR_LSTM_ret, rebalancing_date).values[0]

            # print(f"{rebalancing_date}: {self.new_w(self.compare_var_ret.loc[:rebalancing_date])}")
            # new_w = self.new_w(self.compare_var_ret.loc[:rebalancing_date], rebalancing_date)
            # port_var_ret = (new_w*self.compare_true_ret.loc[rebalancing_date]).sum(axis=1)

            # print(rebalancing_date)
            # print(f"port_MVO_ret: {port_MVO_ret}")
            # print(f"port_var_ret: {port_var_ret}")
            # print(f"port_LSTM_ret: {port_LSTM_ret}")
            # print(f"port_LSTM_X_ret: {port_LSTM_X_ret}")
            # print(f"port_VAR_LSTM_ret: {port_VAR_LSTM_ret}")

            ret_df = ret_df.append(pd.Series({"MVO": port_MVO_ret, "VAR": port_var_ret, "LSTM": port_LSTM_ret,
                                    "LSTM_X": port_LSTM_X_ret, "VAR_LSTM": port_VAR_LSTM_ret}, name=rebalancing_date))
            # ret_df = ret_df.append([])

        # print(ret_df)
        # print(w_MVO_df)
        # print(w_var_df)
        # print(w_LSTM_df)
        # print(w_LSTM_X_df)
        # print(w_VAR_LSTM_df)

        daily_ret_MVO = self.daily_port_ret(w_MVO_df)
        daily_ret_var = self.daily_port_ret(w_var_df)
        daily_ret_LSTM = self.daily_port_ret(w_LSTM_df)
        daily_ret_LSTM_X = self.daily_port_ret(w_LSTM_X_df)
        daily_ret_VAR_LSTM = self.daily_port_ret(w_VAR_LSTM_df)

        init_index = 1000
        ret_df = pd.DataFrame({'MVO': self.daily_port_ret(w_MVO_df),
                               'VAR': self.daily_port_ret(w_var_df),
                               'LSTM': self.daily_port_ret(w_LSTM_df),
                               'LSTM_X': self.daily_port_ret(w_LSTM_X_df),
                               'VAR_LSTM': self.daily_port_ret(w_VAR_LSTM_df)})
        equity_curves_df = (ret_df.copy() + 1)
        equity_curves_df = equity_curves_df.append(pd.DataFrame([[init_index for _ in range(len(equity_curves_df.columns))]],
                                            columns=equity_curves_df.columns,
                                            index=[pd.to_datetime("2015-11-29")])).sort_index().cumprod()
        # print(equity_curves_df)
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True)
        equity_curves_df.plot(ax=ax0, lw=0.7)
        ax0.legend()
        ax0.set_ylabel("Equity Curve")


        ## Rolling Sharpe
        rolling_Sharpe = np.sqrt(252) * (ret_df).rolling(
            126).mean() / ret_df.rolling(126).std()
        rolling_Sharpe.plot(ax=ax1, lw=0.5, legend=False)
        ax1.set_ylabel("126 days Sharpe")

        ## Rolling Max Drawdowns
        def dd(ts):
            return np.min(ts / np.maximum.accumulate(ts)) - 1

        equity_curves_df.rolling(126).apply(dd).plot(ax=ax2, legend=False)
        ax2.set_ylabel("126 days Max Drawdown")

        # plt.show()

        ## Annualised Yearly Sharpe
        ## Need to downsample to yearly basis for annualised Sharpe
        yr_ret_port_df = equity_curves_df.resample('1y').first().pct_change(1)
        annual_std = equity_curves_df.pct_change(252).resample('1y').std()

        annual_Sharpe = (yr_ret_port_df) / annual_std
        print(annual_Sharpe)
        print(annual_Sharpe.mean())

        # print(daily_ret_MVO)
        # print(daily_ret_var)
        # print(daily_ret_LSTM)
        # print(daily_ret_LSTM_X)
        # print(daily_ret_VAR_LSTM)

        print(tabulate(annual_Sharpe.round(2), tablefmt="github", headers=annual_Sharpe.columns))
        print(tabulate(annual_Sharpe.mean().round(2).to_frame().T, headers=annual_Sharpe.columns, tablefmt="github"))


        # ret_MVO = self.mkts_ret*(w_MVO_df).ffill().dropna(axis=0)
        # ret_var = self.mkts_ret.merge(w_var_df, how='left', left_index=True, right_index=True).dropna(axis=0).ffill()
        # print(ret_MVO)
        # print(ret_var)
backtesting(20, long_short=False).main()
