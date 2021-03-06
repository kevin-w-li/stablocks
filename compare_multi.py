import matplotlib as mpl
# mpl.use('Agg')
import sys, io
import pygame
from matplotlib import pyplot as plt
import pygame
from pygame.locals import *
import pymunkoptions

pymunkoptions.options["debug"] = False
import pymunk  # 1
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

plt.rcParams['image.interpolation'] = 'nearest'
display_size = 1000
image_size = 227
all_subjects = []
resp_files = glob.glob("./exp/resp/real/0*")
exps = ['exp_50_5_3', 'exp_50_7_3', 'exp_multi_30']
for fn in resp_files:
    data_dict = load_data(fn)
    # checking whether the data is old one dimensional or multipile and only appending one dimensional data
    if not('exp_30_10_3_space'  in data_dict.keys()):
        all_subjects.append(load_data(fn))
all_mean_resps = deepcopy(all_subjects[0])
all_resps = deepcopy(all_subjects[0])
human_average_given_conf = dict()
model_average_given_conf = dict()
true_average_given_conf = dict()

for exp in exps:
    human_average_given_conf[exp] = []
    model_average_given_conf[exp] = []
    true_average_given_conf[exp] = []

for exp in exps:
    exp_key = exp + '_space'
    for ti, towers in enumerate(all_mean_resps[exp_key]):
        del all_mean_resps[exp_key][ti]['seq']
        all_mean_resps[exp_key][ti] = all_mean_resps[exp_key][ti]['choices']
        for coords in towers['choices']:
            all_mean_resps[exp_key][ti][coords] = 0.0
            all_resps[exp_key][ti][coords] = []
            for subject in all_subjects:
                r = subject[exp_key][ti]['choices'][coords]
                all_mean_resps[exp_key][ti][coords] += r
                all_resps[exp_key][ti][coords].append(r)
            all_mean_resps[exp_key][ti][coords] /= len(all_subjects)
        counter = 0
        temp = 0
        for coords in towers['choices']:
            counter += 1
            temp += all_mean_resps[exp_key][ti][coords]
        human_average_given_conf[exp].append(temp / float(counter))

n_towers = 2
visual = False
if visual:
    fig, axes = plt.subplots(3, n_towers * 2, figsize=(12, 8))

for di, dataset in enumerate(exps):

    space_filename = 'exp/' + dataset + '_space'

    spaces, _ = load_space(space_filename)
    for si, space in enumerate(spaces):
        if visual:
            ax = axes[di, 2 * si]
            plt_options = pymunk.matplotlib_util.DrawOptions(ax)
            ax.set(adjustable='box-forced', aspect=1, xlim=(0, display_size), ylim=(0, display_size))
            plot_space(space, display_size, image_size, fig=fig, ax=ax, plt_options=plt_options)
            # ax.set_axis_off()

            ax = axes[di, 2 * si + 1]
            plt_options = pymunk.matplotlib_util.DrawOptions(ax)
            ax.set(adjustable='box-forced', aspect=1, xlim=(0, display_size), ylim=(0, display_size))
        new_space, _ = copy_space(space)
        block_labels = simulate_whole(space, recog_noise=1., noise_rep=1, det=True)
        # block_labels_arr = np.asarray([value for value in block_labels)
        block_avg = np.mean(np.asarray([block_labels.values()]))
        if visual:
            labeled_data = plot_space_label(space, block_labels, display_size, image_size, fig=fig, ax=ax,
                                            plt_options=plt_options)
        model_average_given_conf[dataset].append(block_avg)
        # ax.set_axis_off()

if visual:
    plt.show()
print human_average_given_conf
print model_average_given_conf
fig, ax = plt.subplots(3, figsize=(3,10))
plt.rcParams.update({'font.size': 10})
plt.rc('text', usetex=True)

for i, exp in enumerate(exps):
    ax[i].scatter(human_average_given_conf[exp], model_average_given_conf[exp], marker='+')
    ax[i].set(adjustable='box-forced', aspect=1, xlim=(0, 1.1), ylim=(0, 1.1))
    ax[i].set_xlabel(r'$\rm{Human}$')
    ax[i].set_ylabel(r'$\rm{Truth}$')
    # ax[i].plot(human_average_given_conf[exp], 'r')
    # ax[i].plot(model_average_given_conf[exp], 'b')
plt.tight_layout()
plt.show()

raise
sim_labels = simulate_whole(spaces[1])

num_blocks = len(spaces)
recog_noise = 5
num_slices = 100

# neural nets
'''
for i in range(5):
    ax = axes[:,i]
    probmap = probmaps[i]
    space = spaces[i]
    plt_options = pymunk.matplotlib_util.DrawOptions(ax[0])
    data = plot_space(space, display_size, image_size, fig, ax[0], plt_options)
    labels = space_array_to_label(space, display_size, probmap)
    labeled_data = plot_space_label(space, labels, display_size, image_size, fig, ax[1], plt_options)
    map(lambda a: a.set_axis_off(), ax)
plt.show()

'''
fig, ax = plt.subplots(1, figsize=(6, 6))
plt_options = pymunk.matplotlib_util.DrawOptions(ax)
ax.set(adjustable='box-forced', aspect=1, xlim=(0, display_size), ylim=(0, display_size))
ax.set_axis_off()


def process_one(n_above, space):
    map(lambda p: p.remove(), filter(lambda c: isinstance(c, mpl.patches.Polygon), ax.get_children()))
    blocks = sort_pile(space.shapes)

    plane_heights, level_labels, det_level_labels = combined_center_of_mass(blocks, recog_noise=recog_noise,
                                                                            n_above=n_above)

    block_labels = labels_to_block_labels(blocks, plane_heights, level_labels)
    return block_labels


from functools import partial

probs = []
for p in [1, 3, 5]:
    process_one_param = partial(process_one, p)
    all_labels = map(process_one_param, spaces)
    probs.append([v for d in all_labels for v in d.values()])

sim_labels = []
det_labels = []
for space in spaces:
    sim_labels.append(simulate_whole(space))
    det_labels.append(simulate_whole(space, det=True))
probs.append([v for d in sim_labels for v in d.values()])
probs.append([v for d in det_labels for v in d.values()])
probs.append([v for d in choices for v in d.values()])
probs = np.array(probs)

print np.corrcoef(probs)



