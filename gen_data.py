import sys, io
import pygame
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl
from pygame.locals import *
from pymunk import pygame_util
from pymunk import matplotlib_util
import pymunk #1
import random
from shapes import *
from dynamics import *
from discrimination import *
import numpy as np
from io_util import *
import multiprocessing

display_size = 600
image_size = 227
my_dpi = 96
block_size = 80
base_width = 5
num_blocks = 5
num_piles = 10
num_slices = 100
plt.rcParams['image.cmap'] = 'gray'

pygame.init()
pygame.display.set_caption("Blocks will fall?")
clock = pygame.time.Clock()
space = pymunk.Space()

space.gravity = (0.0, -100.0)
space.iterations = 5000
space.colission_slope = 0.0

all_data = np.zeros((num_piles, image_size, image_size, 3))
all_slices = np.zeros((num_piles, num_slices))
space = pymunk.Space()

def get_one(i):
    body, shape = make_pile(space, num_of_blocks = num_blocks, base_coord = [(0., 5.), (display_size, 5.)], base_width = base_width,  mass = 1, block_dim = [block_size, block_size/2], noise = 0.4)
    plane_heights, labels = combined_center_of_mass(shape)
    slice_vec = labels_to_strips((0,display_size), plane_heights, labels, num_slices)
    data = space_to_array(space, display_size, image_size)
    return (data, slice_vec, np.any(labels)) 

pool = multiprocessing.Pool(4)
all_data_slices = pool.map(get_one, range(num_piles))
all_data = np.array(map(lambda l:l[0], all_data_slices))
all_slices = np.array(map(lambda l:l[1], all_data_slices))
all_class = np.array(map(lambda l:l[2], all_data_slices))
print 'mean of slices is ', np.mean(all_class)

import h5py
filename = '_'.join(('dataset', str(num_piles), str(num_blocks)))
filename = filename + '.hdf5'
f = h5py.File(filename, 'w')
f.create_dataset('data', data = all_data)
f.create_dataset('label', data = all_slices)
f.close()
plot_pile_slice(all_data[0], all_slices[0])
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