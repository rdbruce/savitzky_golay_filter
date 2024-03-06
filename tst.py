#!/usr/bin/env python3

import math
import yfinance as yf
import savitzky_golay as pf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# stock_data = yf.download('AAPL', start='2021-01-01', interval = "1d")['Close']
# stock_data = stock_data.values

data_len = 1000
stock_data = np.array([100])
for i in range(1, data_len):
    stock_data = np.append(stock_data, stock_data[i - 1]*(1 + 0.1/252 + 0.3*np.random.normal(0, 1)*math.sqrt(1/252)))

smooth_data = pf.savGolFilter(data_in = stock_data, num_elements = stock_data.size, window_size = 7, smoothing_degree = 1, return_degree = 0)
# smooth_data2 = savgol_filter(stock_data, 5, 1)

# print(smooth_data - smooth_data2)
# print(smooth_data)
# print(smooth_data2)

# plt.plot(stock_data, 'g^', smooth_data, 'r-', smooth_data2, 'b-')
plt.plot(stock_data, 'g^', smooth_data, 'r-')
plt.show()