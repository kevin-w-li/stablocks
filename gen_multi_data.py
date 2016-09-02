import matplotlib as mpl
# mpl.use('Agg')
from interval import interval
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

display_size = 1000
image_size = 227
label_size = 50
my_dpi = 96
block_size = 100
base_width = 10
max_num_blocks = 12
min_num_blocks = 6
num_piles = 50
recog_noise = 0
plt.rcParams['image.cmap'] = 'gray'
#assert(block_size * num_blocks < display_size)
pygame.init()
pygame.display.set_caption("Blocks will fall?")
clock = pygame.time.Clock()

space = pymunk.Space()
fig,ax = plt.subplots(1, figsize = (6,6))
plt_options = pymunk.matplotlib_util.DrawOptions(ax)
ax.set(adjustable='box-forced', aspect=1, xlim=(0,display_size), ylim=(0, display_size))
ax.set_axis_off()
def get_one(i):
    space = pymunk.Space()
    map(lambda p: p.remove(), filter(lambda c: isinstance(c, mpl.patches.Polygon), ax.get_children()))
    num_blocks = np.random.randint(min_num_blocks, max_num_blocks)
    num_blocks = np.random.choice([6,8,10])
    '''
    blocks = make_pile(space, num_of_blocks = num_blocks, base_coord = [(0., 5.), (display_size, 5.)], base_width = base_width,  block_dim = [block_size, block_size/2], noise = 0.35)
    '''
    _, blocks = smart_rain_maker(space, num_of_blocks=num_blocks, block_dim = [block_size, block_size/2.0], var = 0.7, base_coord = [(0,5), (display_size, 5)])
    new_space, _ = copy_space(space)
    data = space_to_array(space, display_size, image_size, fig, ax, plt_options)
    block_labels = simulate_whole(space, recog_noise = recog_noise, noise_rep = 1, det = False)
    labeled_data = space_label_to_array(new_space, block_labels, display_size, label_size, fig, ax, plt_options)
    # print (ax.get_children())
    return (data, labeled_data, block_labels)

pool = multiprocessing.Pool(8)
all_data_slices = pool.map(get_one, range(num_piles))
all_data = np.array(map(lambda l:l[0], all_data_slices))
all_labeled_data = np.array(map(lambda l:l[1], all_data_slices))
all_labels = np.array(map(lambda l:l[2], all_data_slices))
all_classes = np.mean(np.array([np.mean(stable.values()) for stable in all_labels]))
print 'mean of class is ', np.mean(all_classes)

filename = '_'.join(('data/dataset', str(num_piles), str(max_num_blocks), str(recog_noise), str(image_size), str(label_size)))
filename = filename + '.hdf5'
f = h5py.File(filename, 'w')
f.create_dataset('data', data = all_data)
f.create_dataset('label', data = all_labeled_data)
f.create_dataset('class', data = all_classes)
f.close()
plot_many_piles_slices(all_data, all_labeled_data)

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
