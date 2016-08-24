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

display_size = 200
image_size = 227
my_dpi = 96
block_size = display_size/10

plt.rcParams['image.cmap'] = 'gray'
pygame.init()
pygame.display.set_caption("Blocks will fall?")
clock = pygame.time.Clock()

num_blocks = 6
space = pymunk.Space()
space.gravity = (0.0, -00.0)
space.iterations = 500
space.colission_slope = 0.0

blocks = [None]*num_blocks

# add the base block, need to figure out how to set positoin...
base = add_base(space, (0,5), (display_size,5), 5)
print base.bb
for bi in range(num_blocks):
    s = add_block(space, 
        position = [(display_size+1)/2+np.random.randn()*0,block_size/2.0+10+block_size*bi], size = block_size)
    blocks[bi] = s
    print bi, s.bb

heights, labels = combined_center_of_mass(blocks)
vec = to_strips((0,display_size), heights, labels, 20)

data = space_to_array(space, display_size, image_size)

screen = pygame.display.set_mode((display_size,display_size))
screen.fill((255,255,255))

draw_options = pygame_util.DrawOptions(screen)
space.debug_draw(draw_options)
pygame.display.flip()
clock.tick(100)

print 'go'
while True:
    space.step(1/50.0)
    screen.fill((255,255,255))
    space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(50)
