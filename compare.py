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

display_size = 300
image_size = 227
data_filename = 'exp/exp_5_5_3_data.hdf5'
space_filename = 'exp/exp_5_5_3_space'
f = h5py.File(data_filename, 'r')
probmaps = np.array(f['labeled_data'])
fig, axes = plt.subplots(2,5, figsize = (12,6))
for i in range(5):
    ax = axes[:,i]
    probmap = probmaps[i]
    spaces, _ = load_space(space_filename)
    space = spaces[i]
    plt_options = pymunk.matplotlib_util.DrawOptions(ax[0])
    data = plot_space(space, display_size, image_size, fig, ax[0], plt_options)
    labels = space_array_to_label(space, display_size, probmap)
    labeled_data = plot_space_label(space, labels, display_size, image_size, fig, ax[1], plt_options)
    map(lambda a: a.set_axis_off(), ax)
plt.show()


