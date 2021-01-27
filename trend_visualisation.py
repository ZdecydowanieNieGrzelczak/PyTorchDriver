import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import random
import pickle


file = open("AzureResults\\Advantages1.p",'rb')
cur = pickle.load(file)

# file = open("forced exp.p",'rb')
# exp = pickle.load(file)

print(np.mean(cur[:200]))
print(np.mean(cur[:-200]))
print(np.shape(cur))

# print("exploration")
#
# print(np.mean(exp[:200]))
# print(np.mean(exp[:-200]))



# MATRIX_SIZE = 1000
# RANDOM_BIAS = 0.3
#
#
# def sliding_window(buffer, window_size=25):
#     new_buffer = []
#     for i in range(len(buffer) - window_size):
#         new_buffer.append(np.sum(buffer[i:i + window_size]) / window_size)
#     return new_buffer
#
#
# data = [1 + i * 0.01 for i in range(MATRIX_SIZE)]
# y = [i for i in range(MATRIX_SIZE)]
#
# for i in range(MATRIX_SIZE):
#     data[i] += data[i] * random.uniform(-RANDOM_BIAS, RANDOM_BIAS)
#
#
# # print(data)
#
#
# plt.plot(y, data, label="raw")
#
# slided = sliding_window(data, 10)
# y10 = [i for i in range(5, MATRIX_SIZE - 5)]
#
# plt.plot(y10, slided, label="avg 10")
#
# slided_50 = sliding_window(data, 50)
# y50 = [i for i in range(25, MATRIX_SIZE - 25)]
#
# plt.plot(y50, slided_50, label="avg 50")
#
# start = data[:50]
# end = data[-50:]
#
# start_value = np.mean(np.asarray(start))
# end_value = np.mean(np.asarray(end))
#
# step = (end_value - start_value) / (MATRIX_SIZE - 50)
#
#
# trend_data = [start_value + i * step for i in range(25, MATRIX_SIZE - 25)]
#
# plt.plot(y50, trend_data, label="trend line")
#
# plt.legend()
#
# plt.savefig("trend visualisation.png")
