import matplotlib as mpl
# mpl.use('Agg')
import sys, io
import pygame
from matplotlib import pyplot as plt
import pygame
from pygame.locals import *
import pymunk #1
from pymunk import pygame_util
from pymunk import matplotlib_util
from shapes import *
from discrimination import *
import numpy as np
from io_util import *
import multiprocessing
import h5py

display_size = 1000
image_size = 227
my_dpi = 96
block_size = 100
base_width = 10
num_blocks = 5
num_piles = 30
recog_noise = 0
pos_noise = 0.8
plt.rcParams['image.cmap'] = 'gray'
pygame.init()
pygame.display.set_caption("Blocks will fall?")
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0.0, -100.0)
space.iterations = 5000
space.colission_slope = 0.0

all_data = np.zeros((num_piles, image_size, image_size, 3), dtype=np.uint8)

fig,ax = plt.subplots(1, figsize = (6,6))
plt_options = pymunk.matplotlib_util.DrawOptions(ax)
ax.set(adjustable='box-forced', aspect=1, xlim=(0,display_size), ylim=(0, display_size))
ax.set_axis_off()


def get_one(i):
    space = pymunk.Space()
    map(lambda p: p.remove(), filter(lambda c: isinstance(c, mpl.patches.Polygon), ax.get_children()))
    num_blocks = 6+(i/10)*2 
    '''
    blocks = make_pile(space, num_of_blocks = num_blocks, base_coord = [(0., 5.), (display_size, 5.)], base_width = base_width,  block_dim = [block_size, block_size/2], noise = 0.35)
    '''
    _, blocks = smart_rain_maker(space, num_of_blocks=num_blocks, block_dim = [block_size, block_size/2.0], var = pos_noise, base_coord = [(0,5), (display_size, 5)])
    new_space, _ = copy_space(space)
    data = space_to_array(space, display_size, image_size, fig, ax, plt_options)
    block_labels = simulate_whole(space, recog_noise = recog_noise, noise_rep = 1, det = True)
    labeled_data = space_label_to_array(new_space,block_labels, display_size, image_size, fig, ax, plt_options)
    # print (ax.get_children())
    return (data, labeled_data, block_labels, space)

all_data_slices = map(get_one, range(num_piles))
all_data = np.array(map(lambda l:l[0], all_data_slices))
all_labeled_data = np.array(map(lambda l:l[1], all_data_slices))
all_block_labels = map(lambda l:l[2], all_data_slices)
all_classes = np.array([all(stable.values()) for stable in all_block_labels])
all_spaces = np.array(map(lambda l:l[3], all_data_slices))

print 'mean of class is ', np.mean(all_classes)
filename = '_'.join(('exp/exp_multi', str(num_piles), 'data'))
filename = filename + '.hdf5'
f = h5py.File(filename, 'w')
f.create_dataset('data', data = all_data)
f.create_dataset('labeled_data', data = all_labeled_data)
f.create_dataset('class', data = all_classes)
f.close()
save_space(all_spaces, all_block_labels, \
    '_'.join(('exp/exp_multi', str(num_piles), 'space')))
plot_many_piles_slices(all_data, all_labeled_data)
