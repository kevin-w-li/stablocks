import sys
import pygame
from pygame.locals import *
import pymunk #1
import random
from shapes import add_block, add_base, make_pile_given_noise
from scipy.stats import truncnorm, uniform
import time
import numpy as np

pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Blocks will fall?")
clock = pygame.time.Clock()


blocks = []
from pymunk import pygame_util

# add the base block, need to figure out how to set positoin...
# base = add_base(space, (0,100), (500,100))

num_of_blocks = 3
noise_trials = 3
block_arrangements_num = 1
var = 1.
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit(0)
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            sys.exit(0)

    # ticks_to_next_block -= 1
    #
    # # add a block every 25 ticks
    # if ticks_to_next_block <= 0:
    #
    #     ticks_to_next_block = 25
    #     s = add_block(space, position = [300,500], size = 50)
    #     blocks.append(s)

    for i in range(block_arrangements_num):
        position_noise_list = truncnorm.rvs(- 1. / (var), 1. / (var), size=num_of_blocks) * var
        hor_ver_list = np.random.binomial(2, 0.5, num_of_blocks)
        for j in range(num_of_blocks):
            for k in range(noise_trials):
                noise = np.random.randn(var)
                space = pymunk.Space()
                space.gravity = (0.0, -900.0)
                draw_options = pygame_util.DrawOptions(screen)
                print i, j
                b_list, s_list = make_pile_given_noise(space=space, noise_list=position_noise_list,
                                                       hor_ver_list=hor_ver_list)
                for step_num in range(100):
                    screen.fill((255, 255, 255))
                    space.debug_draw(draw_options)
                    if step_num == 20:
                        b_list[j].apply_force_at_local_point([noise, 0.], b_list[j].position)
                    space.step(1 / 50.0)
                    clock.tick(50)
                    pygame.display.flip()
                for k in range(len(s_list)):
                    space.remove(b_list[0])
                    space.remove(s_list[0])
                    b_list.remove(b_list[0])
                    s_list.remove(s_list[0])





