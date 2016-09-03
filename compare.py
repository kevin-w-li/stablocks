import matplotlib as mpl
# mpl.use('Agg')
import sys, io
import pygame
from matplotlib import pyplot as plt
import pygame
from pygame.locals import *
import pymunkoptions
pymunkoptions.options["debug"] = False
import pymunk #1
from pymunk import pygame_util
from pymunk import matplotlib_util
from shapes import *
from discrimination import *
import numpy as np
from io_util import *
import multiprocessing
import h5py
from io_util import *
import glob
from copy import deepcopy
import os
# plt.rcParams['image.interpolation'] = 'nearest'
display_size = 1000
image_size = 1000
label_size = 1000
sim_block_labels = pkl.load(open('exp/sim_block_labels'))
probmaps = np.load(open('exp/nn_probmaps'))

all_subjects = []
resp_files = glob.glob("./exp/resp/real/0*")
exps = ['exp_50_5_3', 'exp_50_7_3', 'exp_30_10_3' ]

for fn in resp_files:
    data_dict = load_data(fn)
    # checking whether the data is old one dimensional or multipile and only appending one dimensional data
    if 'exp_30_10_3' in data_dict.keys():
        resps = load_data(fn)
        del resps['exp_20_4_3']
        all_subjects.append(resps)

all_mean_resps = deepcopy(all_subjects[0])
all_resps = deepcopy(all_subjects[0])

human_average_given_conf = dict()
truth_average_given_conf = dict()
npe_average_given_conf = dict()
nn_average_given_conf = dict()



for exp in exps:
    human_average_given_conf[exp] = []
    truth_average_given_conf[exp] = []
for exp in exps:
    for ti, towers in enumerate(all_mean_resps[exp]):
        del all_mean_resps[exp][ti]['seq']
        all_mean_resps[exp][ti] = all_mean_resps[exp][ti]['choices']
        for coords in towers['choices']:
            all_mean_resps[exp][ti][coords] = 0.0
            all_resps[exp][ti][coords] = []
            for subject in all_subjects:
                r = subject[exp][ti]['choices'][coords]
                all_mean_resps[exp][ti][coords] += r
                all_resps[exp][ti][coords].append(r)
            all_mean_resps[exp][ti][coords] /= len(all_subjects)
        counter = 0
        temp = 0
        for coords in towers['choices']:
            counter += 1
            temp += all_mean_resps[exp][ti][coords]
        human_average_given_conf[exp].append(temp / float(counter))
# define function that takes responses and compute average number of blocks to fall
def num_falling_blocks(r):
    average = dict()
    for exp in exps:
        average[exp] = [np.mean(t.values()) for t in r[exp]]
    return average

n_towers = 2
visual = False
# fig, axes = plt.subplots(5,n_towers*3, figsize = (16, 8))

all_true_resps = deepcopy(all_mean_resps)
all_nn_resps = deepcopy(all_mean_resps)
all_npe_resps = deepcopy(all_mean_resps)
all_human_resps = all_mean_resps 

for di, exp in enumerate(exps):
    
    space_filename = 'exp/'+exp+'_space'
    spaces, _ = load_space(space_filename)
    for si, space in enumerate(spaces):
        print [di, si] 
        # ground truth data
        true_labels = sim_block_labels[exp][si]
        # data, true_labeled_data = space_label_to_array(space, true_labels, display_size, image_size, label_size)
        # block_avg = np.mean(np.asarray([true_labels.values()]))
        # truth_average_given_conf[exp].append(block_avg)
        all_true_resps[exp][si] = true_labels

        human_labels = all_mean_resps[exp][si]
        # human_labeled_data = space_label_to_array(space, human_labels, display_size, image_size, label_size)[1]
        npe_labels = simulate_whole(space, recog_noise = 20., noise_rep = 2, det = False)
        # npe_labeled_data = space_label_to_array(space, npe_labels, display_size, image_size, label_size)[1]
        all_npe_resps[exp][si] = npe_labels

        probmap = probmaps[exp][si]
        nn_labels = space_array_to_label(space, display_size, probmap)
        # nn_labeled_data = space_label_to_array(space, nn_labels, display_size, image_size, label_size)[1]
        all_nn_resps[exp][si] = nn_labels

        '''
        if si<n_towers:
            # tower
            ax = axes[0, n_towers*di+si]
            ax.set(adjustable='box-forced', aspect=1, xlim=(0+300,display_size-300), ylim=(0, display_size))
            ax.imshow(data); ax.invert_yaxis(); ax.set_axis_off(); 
            if si == 0: ax.set_ylabel('tower')

            # truth
            ax = axes[1, n_towers*di+si]
            ax.set(adjustable='box-forced', aspect=1, xlim=(0+300,display_size-300), ylim=(0, display_size))
            ax.imshow(true_labeled_data, cmap = 'Blues_r'); ax.invert_yaxis(); ax.set_axis_off()
            if si == 0: ax.set_ylabel('truth')


            # human
            ax = axes[2, n_towers*di+si]
            ax.set(adjustable='box-forced', aspect=1, xlim=(0+300,display_size-300), ylim=(0, display_size))
            ax.imshow(human_labeled_data, cmap = 'Greens_r'); ax.invert_yaxis(); ax.set_axis_off()
            if si == 0: ax.set_ylabel('human')

            # noisy PE
            ax = axes[3, n_towers*di+si]
            ax.set(adjustable='box-forced', aspect=1, xlim=(0+300,display_size-300), ylim=(0, display_size))
            ax.imshow(npe_labeled_data, cmap = 'Reds_r'); ax.invert_yaxis(); ax.set_axis_off()
            if si == 0: ax.set_ylabel('IPE')

            # nn  
            ax = axes[4, n_towers*di+si]
            ax.set(adjustable='box-forced', aspect=1, xlim=(0+300,display_size-300), ylim=(0, display_size))
            ax.imshow(nn_labeled_data, cmap = 'bone'); ax.invert_yaxis(); ax.set_axis_off()
            if si == 0: ax.set_ylabel('NN')

        '''
        # ax.set_axis_off()

human_average_given_conf = num_falling_blocks(all_human_resps)
truth_average_given_conf = num_falling_blocks(all_true_resps)
npe_average_given_conf = num_falling_blocks(all_npe_resps)
nn_average_given_conf = num_falling_blocks(all_nn_resps)
if visual:
    plt.show()
fig, axes = plt.subplots(3,3)
for i, exp in enumerate(exps):
    n = len(human_average_given_conf[exp])
    corr_true = np.corrcoef(human_average_given_conf[exp], truth_average_given_conf[exp])[0,1]
    corr_npe = np.corrcoef(human_average_given_conf[exp], npe_average_given_conf[exp])[0,1]
    corr_nn = np.corrcoef(human_average_given_conf[exp], nn_average_given_conf[exp])[0,1]
    axes[0,i].scatter(human_average_given_conf[exp], truth_average_given_conf[exp], c='g')
    axes[1,i].scatter(human_average_given_conf[exp], npe_average_given_conf[exp], c='r')
    axes[2,i].scatter(human_average_given_conf[exp], nn_average_given_conf[exp], c = np.tile([227, 218, 201],(n,1))/255.)
    '''
    ax[i].plot(human_average_given_conf[exp], 'r')
    ax[i].plot(truth_average_given_conf[exp], 'b')
    '''
map(lambda ax: map(lambda a: a.set(adjustable='box-forced', aspect=1, xlim=(0, 1), ylim=(0,1)), ax), axes)

plt.show()

num_blocks = len(spaces)
recog_noise = 5
num_slices = 100
'''

exps = ['exp_20_4_3', 'exp_50_5_3', 'exp_50_7_3', 'exp_30_10_3' ]
exps_keys = [s+'_space' for s in exps]
# for di, exp in enumerate(exps):
exp = exps[0]
space_filename = 'exp/'+exp+'_space'
spaces, _ = load_space(space_filename)
probmaps = np.load(open('exp/nn_probmaps'))
print probmaps
fig, axes = plt.subplots(3,10)
for i in range(10):
    ax = axes[:,i]
    probmap = probmaps['exp_50_5_3'][i]
    space = spaces[i]
    labels = space_array_to_label(space, display_size, probmap)
    data = space_to_array(space, display_size, image_size)
    ax[0].imshow(data)
    labeled_data = space_label_to_array(space, labels, display_size, image_size, label_size)[1]
    map(lambda a: a.set_axis_off(), ax)
    ax[1].imshow(probmap, cmap = 'gray')
    ax[2].imshow(labeled_data, cmap = 'gray')
plt.show()

fig,ax = plt.subplots(1, figsize = (6,6))
plt_options = pymunk.matplotlib_util.DrawOptions(ax)
ax.set(adjustable='box-forced', aspect=1, xlim=(0,display_size), ylim=(0, display_size))
ax.set_axis_off()

def process_one(n_above, space):
    map(lambda p: p.remove(), filter(lambda c: isinstance(c, mpl.patches.Polygon), ax.get_children()))
    blocks = sort_pile(space.shapes)

    plane_heights, level_labels, det_level_labels = combined_center_of_mass(blocks, recog_noise = recog_noise, n_above = n_above)
    
    block_labels = labels_to_block_labels(blocks, plane_heights, level_labels)
    return block_labels

from functools import partial
probs = []
for p in [1,3,5]:
    process_one_param = partial(process_one, p)
    all_labels = map(process_one_param, spaces)
    probs.append([v for d in all_labels for v in d.values()])
print "CoG done"
rand_labels = []
for space in spaces:
    rand_labels.append(simulate_whole(space))
probs.append([v for d in rand_labels for v in d.values()])
probs.append([v for d in sim_block_labels for v in d.values()])
probs.append([v for d in all_mean_resps for v in d.values()])
probs = np.array(probs)

print np.corrcoef(probs)

'''
