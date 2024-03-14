#!/usr/bin/env python3

import math
import yfinance as yf
import savitzky_golay as pf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time

# stock_data = yf.download('AAPL', start='2021-01-01', interval = "1d")['Close']
# stock_data = stock_data.values

data_len = 25
stock_data = np.array([100])
for i in range(1, data_len):
    stock_data = np.append(stock_data, stock_data[i - 1] * (1 + 0.1/252 + 0.10 * math.sqrt(1/252) * np.random.default_rng().normal(0, 1)))
# print(stock_data.size)
    
tic = time.perf_counter()
sg = pf.SavGol(data_in = stock_data, num_elements = stock_data.size, window_size = 5, smoothing_degree = 1, return_degree =0, threaded = False)
smooth_data = sg.filter()
toc = time.perf_counter()
nonThread = toc - tic

tic = time.perf_counter()
sg = pf.SavGol(data_in = stock_data, num_elements = stock_data.size, window_size = 5, smoothing_degree = 1, return_degree =0, threaded = True)
smooth_data = sg.filter()
toc = time.perf_counter()
thread = toc - tic

print(f"My function ran in {nonThread:0.10f} seconds")
print(f"My threaded function ran in {thread:0.10f} seconds")

# print(smooth_data)

tic = time.perf_counter()
smooth_data2 = savgol_filter(stock_data, 5, 1, 0)
toc = time.perf_counter()
print(f"Their function ran in {toc - tic:0.10f} seconds")

# print(smooth_data - smooth_data2)
# print(smooth_data)
# print(smooth_data2)

plt.plot(stock_data, 'g^', smooth_data, 'r-', smooth_data2, 'b-')
# plt.plot(stock_data, 'g^', smooth_data, 'r-')
plt.show()