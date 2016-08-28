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

def pile_stability_w_noise(num_of_blocks=7, noise_trials=5, block_arrangements_num=3, var=1.):
    for i in range(block_arrangements_num):
        position_noise_list = truncnorm.rvs(- 1. / (var), 1. / (var), size=num_of_blocks)
        hor_ver_list = np.random.binomial(2, 0.5, num_of_blocks)
        print hor_ver_list
        for j in range(num_of_blocks):
            for k in range(noise_trials):

                # just for stopping the code from running
                for event in pygame.event.get():
                    if event.type == QUIT:
                        sys.exit(0)
                    elif event.type == KEYDOWN and event.key == K_ESCAPE:
                        sys.exit(0)

                noise = np.random.randn(1) * 0
                space = pymunk.Space()
                space.gravity = (0.0, -900.0)
                draw_options = pygame_util.DrawOptions(screen)
                print i, j
                b_list, s_list = make_pile_given_noise(space=space, position_noise_list=position_noise_list,
                                                       hor_ver_list=hor_ver_list)
                for step_num in range(200):
                    screen.fill((255, 255, 255))
                    space.debug_draw(draw_options)
                    if step_num == 100:
                        b_list[j].apply_force_at_local_point([noise, 0.], b_list[j].position)
                    space.step(1 / 50.0)
                    clock.tick(50)
                    pygame.display.flip()
                for k in range(len(s_list)):
                    space.remove(b_list[0])
                    space.remove(s_list[0])
                    b_list.remove(b_list[0])
                    s_list.remove(s_list[0])

while True:
    pile_stability_w_noise(num_of_blocks=7, noise_trials=5, block_arrangements_num=3, var=1)





