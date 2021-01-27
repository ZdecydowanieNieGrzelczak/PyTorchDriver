import torch
import numpy as np
import time
import pickle
import matplotlib
from matplotlib import pyplot as plt
from ConvEnv import ConvEnv

# def sliding_window(buffer, window_size=25):
#     new_buffer = []
#     for i in range(len(buffer) - window_size):
#         new_buffer.append(np.sum(buffer[i:i + window_size]) / window_size * 100)
#     return new_buffer
#
#
# files = []
# for i in range(11):
#     buffer = pickle.load(open("AzureResults\\Advantages" + str(i + 1) + ".p", "rb" ))
#     print(i, np.shape(buffer))
#     files += buffer
#
# print(np.shape(files))
#
#
# print(files[0])
# reward1 = pickle.load( open( "AzureResults\\Rewards1.p", "rb" ) )
# reward2 = pickle.load( open( "AzureResults\\Rewards2.p", "rb" ) )
# reward3 = pickle.load( open( "AzureResults\\Rewards3.p", "rb" ) )
# reward4 = pickle.load( open( "AzureResults\\Rewards4.p", "rb" ) )
# reward5 = pickle.load( open( "AzureResults\\Rewards5.p", "rb" ) )
# reward6 = pickle.load( open( "AzureResults\\Rewards6.p", "rb" ) )
# reward7 = pickle.load( open( "AzureResults\\Rewards7.p", "rb" ) )
# reward8 = pickle.load( open( "AzureResults\\Rewards8.p", "rb" ) )
# reward9 = pickle.load( open( "AzureResults\\Rewards9.p", "rb" ) )
# reward10 = pickle.load( open( "AzureResults\\Rewards10.p", "rb" ) )
# reward11 = pickle.load( open( "AzureResults\\Rewards11.p", "rb" ) )

actor_loss = pickle.load( open( "Graphs\\Actor loss.p"))

print(np.shape(actor_loss))

#
# print("loaded")
#
#
#
# slided_buffer = sliding_window(files, 10000)
#
#
#
# X = [i for i in range(len(slided_buffer))]
#
# plt.plot(X, slided_buffer)
#
# # plt.show()
#
# plt.savefig("final_results.png")

# env = ConvEnv(quest_nr=4, station_nr=6, width=10, height=10, uniform_gas_stations=True, normalize_rewards=True)
#
# env.player_pos = [6, 2]
#
# env.get_state_object()