from io_util import load_data
import os
import glob
from matplotlib import pyplot as plt

file_list = os.chdir('./data/')
reaction_time = dict()
keys = ['exp_20_4_3_space', 'exp_50_5_3_space', 'exp_50_7_3_space', 'exp_30_10_3_space']
def reaction_times_hist():
    for key in keys:
        reaction_time[key] = []
    for file_name in glob.glob("*"):
        data = load_data(file_name)
        print data.keys()
        for key in data.keys():
            for i in range(len(data[key]) - 1):
                reaction_time[key].append(data[key][i + 1]['seq'][0][0] - data[key][i]['seq'][0][0])
    exp_nums = len(data.keys())
    fig, ax = plt.subplots(exp_nums)
    for i in range(exp_nums):
        ax[i].hist(reaction_time[data.keys()[i]], bins = 40)
        ax[i].set_xlabel('Reaction Time(ms)')
    plt.show()
reaction_times_hist()