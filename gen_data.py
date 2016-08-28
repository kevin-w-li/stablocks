import matplotlib as mpl
mpl.use('Agg')
import sys, io
import pygame
from matplotlib import pyplot as plt
from pygame.locals import *
from pymunk import pygame_util
from pymunk import matplotlib_util
import pymunk #1
from shapes import *
from discrimination import *
import numpy as np
from io_util import *
import multiprocessing
import h5py

display_size = 600
image_size = 227
my_dpi = 96
block_size = 60
base_width = 5
num_blocks = 5
num_piles = 11
num_slices = 50
recog_noise = 5
plt.rcParams['image.cmap'] = 'gray'
assert(block_size * num_blocks < display_size)
pygame.init()
pygame.display.set_caption("Blocks will fall?")
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0.0, -100.0)
space.iterations = 5000
space.colission_slope = 0.0

all_data = np.zeros((num_piles, image_size, image_size, 3), dtype=np.uint8)
all_slices = np.zeros((num_piles, num_slices))

fig,ax = plt.subplots(1, figsize = (6,6))
plt_options = pymunk.matplotlib_util.DrawOptions(ax)
ax.set(adjustable='box-forced', aspect=1, xlim=(0,display_size), ylim=(0, display_size))
ax.set_axis_off()
def get_one(i):
    space = pymunk.Space()
    map(lambda p: p.remove(), filter(lambda c: isinstance(c, mpl.patches.Polygon), ax.get_children()))
    body, shape = make_pile(space, num_of_blocks = num_blocks, base_coord = [(0., 5.), (display_size, 5.)], base_width = base_width,  block_dim = [block_size, block_size/2], noise = 0.5)
    plane_heights, labels = combined_center_of_mass(shape, recog_noise = recog_noise)
    slice_vec = labels_to_strips((0,display_size), plane_heights, labels, num_slices)
    data = space_to_array(space, display_size, image_size, fig, ax, plt_options)
    # print (ax.get_children())
    return (data, slice_vec, all(np.array(labels)>0.5))

pool = multiprocessing.Pool(2)
all_data_slices = pool.map(get_one, range(num_piles))
all_data = np.array(map(lambda l:l[0], all_data_slices))
all_slices = np.array(map(lambda l:l[1], all_data_slices))
all_class = np.array(map(lambda l:l[2], all_data_slices))

print 'mean of class is ', np.mean(all_class)
filename = '_'.join(('data/dataset', str(num_piles), str(num_blocks), str(recog_noise)))
filename = filename + '.hdf5'
f = h5py.File(filename, 'w')
f.create_dataset('data', data = all_data)
f.create_dataset('label', data = all_slices)
f.close()
plot_many_piles_slices(all_data,all_slices)

'''
screen = pygame.display.set_mode((display_size,display_size))
screen.fill((255,255,255))

draw_options = pygame_util.DrawOptions(screen)
space.debug_draw(draw_options)
print 'go'
while True:
    space.step(1/50.0)
    screen.fill((255,255,255))
    space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(50)
'''
