import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import pymunkoptions
pymunkoptions.options["debug"] = False
import pymunk #1
from pymunk import matplotlib_util
from shapes import *
from discrimination import *
import numpy as np
import multiprocessing
import h5py
from io_util import *
import glob
from copy import deepcopy
import os
display_size = 1000
image_size = 227
label_size = 100
exps = ['exp_20_4_3', 'exp_50_5_3', 'exp_50_7_3', 'exp_30_10_3' ]

space = pymunk.Space()
fig,ax = plt.subplots(1, figsize = (6,6))
plt_options = pymunk.matplotlib_util.DrawOptions(ax)
ax.set(adjustable='box-forced', aspect=1, xlim=(0,display_size), ylim=(0, display_size))
ax.set_axis_off()
data = []
labeled_data = []
block_labels = OrderedDict(zip(exps, [[]]*len(exps)))
for di, dataset in enumerate(exps):
    
    space_filename = 'exp/'+dataset+'_space'
    spaces, _ = load_space(space_filename)
    num_towers = len(spaces)
    for si, space in enumerate(spaces):
        print [di, si]
        data.append(space_to_array(space, display_size, image_size, fig, ax, plt_options))
        ax.clear()
        block_labels[dataset].append(simulate_whole(space, recog_noise = 1., noise_rep = 1, det = True))
        labeled_data.append(space_label_to_array(space, block_labels[dataset][si], display_size, label_size, \
                            fig=fig, ax=ax, plt_options = plt_options))
        ax.clear()
        '''
        _, axes = plt.subplots(2)
        axes[0].imshow(data[si])
        axes[1].imshow(labeled_data[si])
        plt.show()
        '''
data = np.array(data)
print data.shape
labeled_data = np.array(labeled_data)
print labeled_data.shape
print block_labels
labeled_data = np.float32(labeled_data>0.95)
filename = 'exp/sim_data_label'
filename = filename + '.hdf5'
f = h5py.File(filename, 'w')
f.create_dataset('data', data = data)
f.create_dataset('label', data = labeled_data)

pkl.dump(labeled_data, open('exp/sim_block_labels', 'w'))
