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

display_size = 1000
image_size = 227
dataset = 'exp_50_5_3'
data_filename = 'exp/'+dataset+'_data.hdf5'
space_filename = 'exp/'+dataset+'_space'
resp_filename = 'exp/resp/real/09-01-20-22'


resps = pkl.load(open(resp_filename))
resp = resps[dataset+'_space']
choices = [t['choices'] for t in resp]
for ci,c in enumerate(choices):
    for bi, v in c.items():
        choices[ci][bi] = float(v)
    
f = h5py.File(data_filename, 'r')
probmaps = np.array(f['labeled_data'])
spaces, _ = load_space(space_filename)
plot_space(spaces[1], display_size, image_size)
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

sim_labels = []
det_labels = []
for space in spaces:
    sim_labels.append(simulate_whole(space))
    det_labels.append(simulate_whole(space, det = True))
probs.append([v for d in sim_labels for v in d.values()])
probs.append([v for d in det_labels for v in d.values()])
probs.append([v for d in choices for v in d.values()])
probs = np.array(probs)

print np.corrcoef(probs)



