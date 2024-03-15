import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

data_path = 'durhamtemp_1901_2019.csv'
data = pd.read_csv(data_path)

data.dropna(how='all', inplace=True)

if 'Date' not in data.columns:
    data.reset_index(inplace=True)

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

data.set_index('Date', inplace=True)

data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day

data['DayOfYear'] = data.index.dayofyear

data['WeekOfYear'] = data.index.isocalendar().week

data['DayOfWeek'] = data.index.dayofweek

data['Quarter'] = data.index.quarter

data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)

data['Tmax_lag1'] = data['Tmax'].shift(1)
data['Tmax_lag2'] = data['Tmax'].shift(2)
data['Tmax_lag3'] = data['Tmax'].shift(3)
data['Tmin_lag1'] = data['Tmin'].shift(1)
data['Tmin_lag2'] = data['Tmin'].shift(2)
data['Tmin_lag3'] = data['Tmin'].shift(3)
data['Av_temp_lag1'] = data['Av temp'].shift(1)
data['Av_temp_lag2'] = data['Av temp'].shift(2)
data['Av_temp_lag3'] = data['Av temp'].shift(3)

data['Tmax_rolling_mean_3'] = data['Tmax'].rolling(window=3).mean()
data['Tmax_rolling_mean_7'] = data['Tmax'].rolling(window=7).mean()
data['Tmin_rolling_mean_3'] = data['Tmin'].rolling(window=3).mean()
data['Tmin_rolling_mean_7'] = data['Tmin'].rolling(window=7).mean()
data['Av_temp_rolling_mean_3'] = data['Av temp'].rolling(window=3).mean()
data['Av_temp_rolling_mean_7'] = data['Av temp'].rolling(window=7).mean()

data['Tmax_max'] = data['Tmax'].rolling(window=30).max()
data['Tmax_min'] = data['Tmax'].rolling(window=30).min()
data['Tmax_mean'] = data['Tmax'].rolling(window=30).mean()
data['Tmax_median'] = data['Tmax'].rolling(window=30).median()
data['Tmax_std'] = data['Tmax'].rolling(window=30).std()

data['Tmin_max'] = data['Tmin'].rolling(window=30).max()
data['Tmin_min'] = data['Tmin'].rolling(window=30).min()
data['Tmin_mean'] = data['Tmin'].rolling(window=30).mean()
data['Tmin_median'] = data['Tmin'].rolling(window=30).median()
data['Tmin_std'] = data['Tmin'].rolling(window=30).std()

data['Av_temp_max'] = data['Av temp'].rolling(window=30).max()
data['Av_temp_min'] = data['Av temp'].rolling(window=30).min()
data['Av_temp_mean'] = data['Av temp'].rolling(window=30).mean()
data['Av_temp_median'] = data['Av temp'].rolling(window=30).median()
data['Av_temp_std'] = data['Av temp'].rolling(window=30).std()

data['Temp_diff'] = data['Tmax'] - data['Tmin']
data['Tmax_diff'] = data['Tmax'] - data['Av temp']
data['Tmin_diff'] = data['Tmin'] - data['Av temp']

data['temp'] = data['Av temp']

diff_seasonal = data['temp'].diff(periods=12)
diff_seasonal = diff_seasonal.dropna()

diff_first = diff_seasonal.diff(periods=1)
diff_first = diff_first.dropna()


def evaluate_model(train_data, order, seasonal_order):
    scale_factor = 100
    train_data_scaled = train_data * scale_factor
    test_data = train_data[-12:]
    test_data_scaled = test_data * scale_factor

    model = SARIMAX(train_data_scaled[:-12], order=order, seasonal_order=seasonal_order, initialization='approximate_diffuse', freq='D')
    results = model.fit()
    predictions = results.forecast(steps=len(test_data)) / scale_factor
    mse = mean_squared_error(test_data, predictions)
    return mse

def grid_search(data, order_params, seasonal_params, s):
    best_mse = float('inf')
    best_params = None

    for order, seasonal_order in itertools.product(order_params, seasonal_params):
        mse_scores = []
        for train_index, _ in tscv.split(data):
            train_data = data[train_index]
            try:
                mse = evaluate_model(train_data, order, seasonal_order + (s,))
                mse_scores.append(mse)
            except:
                continue
        if len(mse_scores) > 0:
            avg_mse = sum(mse_scores) / len(mse_scores)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = (order, seasonal_order)

    return best_mse, best_params

if __name__ == "__main__":

    p = range(0, 3)
    d = range(0, 2)
    q = range(0, 3)
    P = range(0, 2)
    D = range(0, 1)
    Q = range(0, 2)
    s = 12

    # Generate all possible combinations of parameters
    order_params = list(itertools.product(p, d, q))
    seasonal_params = list(itertools.product(P, D, Q))

    # Define the time series cross-validation splitter
    tscv = TimeSeriesSplit(n_splits=5)

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = []
        for train_index, _ in tscv.split(data['temp']):
            train_data = data['temp'][train_index]
            future = executor.submit(grid_search, train_data, order_params, seasonal_params, s)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            mse, params = future.result()
            if params is not None:
                results.append((mse, params))

    if results:
        best_mse, best_params = min(results, key=lambda x: x[0])
        print(f"Best parameters: SARIMA{best_params[0]} x {best_params[1]+(s,)}")
        print(f"Best MSE: {best_mse:.2f}")
    else:
        print("No valid model found.")
