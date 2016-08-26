import sys
import pygame
from pygame.locals import *
import pymunk #1
import random
from shapes import add_block, add_base, make_pile_given_noise, apply_noise
from scipy.stats import truncnorm, uniform
import time


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

    screen.fill((255,255,255))
    for i in range(block_arrangements_num):
        for j in range(noise_trials):
            space = pymunk.Space()
            space.gravity = (0.0, -900.0)
            space.step(1 / 50.0)
            draw_options = pygame_util.DrawOptions(screen)
            noise_list = truncnorm.rvs(- 1. / (var), 1. / (var), size = num_of_blocks) * var
            s_list, b_list = make_pile_given_noise(space=space, noise_list=noise_list)
            apply_noise(s_list, num_of_trials=3)
            space.debug_draw(draw_options)
            for k in range(len(s_list)):
                space.remove(b_list[0])
                space.remove(s_list[0])
                b_list.remove(b_list[0])
                s_list.remove(s_list[0])
            del s_list
            del b_list
            pygame.display.flip()
    clock.tick(50)





