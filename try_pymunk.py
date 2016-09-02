import sys
import pygame
from pygame.locals import *
import pymunk #1
import random
from shapes import add_block, add_base, make_pile_given_noise
from scipy.stats import truncnorm, uniform
import time
import numpy as np
from interval import interval
import copy
import multiprocessing
visual = False
parallel = False
import functools
import cPickle as pickle

if visual:
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Blocks will fall?")
    clock = pygame.time.Clock()


blocks = []
from pymunk import pygame_util

# add the base block, need to figure out how to set positoin...
# base = add_base(space, (0,100), (500,100))

def pile_stability_w_noise(num_of_blocks=7, noise_trials=5, block_arrangements_num=3, var=1.):
    """ Deprecated NOT IN USE!!!"""
    position_noise_list = truncnorm.rvs(- 1. / (var), 1. / (var), size=num_of_blocks)
    hor_ver_list = np.random.binomial(2, 0.5, num_of_blocks)
    for j in range(num_of_blocks):
        for k in range(noise_trials):
            if visual:
                # just for stopping the code from running
                for event in pygame.event.get():
                    if event.type == QUIT:
                        sys.exit(0)
                    elif event.type == KEYDOWN and event.key == K_ESCAPE:
                        sys.exit(0)

            noise = np.random.randn(1) * 0
            space = pymunk.Space()
            space.gravity = (0.0, -900.0)
            if visual:
                draw_options = pygame_util.DrawOptions(screen)
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

def rain_maker(num_of_blocks=20, block_dim = [100,40], var=1.,  base_coord = [(0., 100.), (600., 100.)], base_width = 10, mass=0.001):
    """ Old rain_maker with dropping objects. Do not use."""
    position_noise_list = truncnorm.rvs(- 1. / (var), 1. / (var), size=num_of_blocks)
    hor_ver_list = np.random.binomial(2, 0.5, num_of_blocks)
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    add_base(space, base_coord[0], base_coord[1], width=base_width)
    body_list=[]
    for j in range(num_of_blocks):
        perm_block_dim = block_dim[:]
        if hor_ver_list[j]:
            perm_block_dim[0], perm_block_dim[1] = block_dim[1], block_dim[0]
        block_pos = [position_noise_list[j] * 150 + 300, 600]
        body, shape = add_block(space, block_pos, mass=mass, block_dim=perm_block_dim)
        body_list.append(body)
        f = 100.0
        body.apply_force_at_local_point([0., -f], [0,0])
        if visual:
            # just for stopping the code from running
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)

        draw_options = pygame_util.DrawOptions(screen)
        # while body.velocity <=1:
        #     print body.force[0]

        while body.force[1] == -f:
            if visual:
                screen.fill((255, 255, 255))
                space.debug_draw(draw_options)
                body.apply_force_at_local_point([0., -f], [0.,0.])
                pygame.display.flip()
        clock.tick(50)
        space.step(1 / 100.0)
        for body in body_list:
            body._set_angular_velocity(0)
            body._set_velocity([0,0])
            body._set_force([0,0])
            body._set_torque(0)
    if visual:
        while(True):
            # just for stopping the code from running
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)

def smart_rain_maker(space, num_of_blocks=5, block_dim=[100, 40], block_arrangements_num=3, var=1.,
               base_coord=[(0., 100.), (600., 100.)], base_width=10, mass=0.01, position_noise_list=None, hor_ver_list = None):
    if position_noise_list is None:
        position_noise_list = truncnorm.rvs(- 1. / (var), 1. / (var), size=num_of_blocks)
    if hor_ver_list is None:
        hor_ver_list = np.random.binomial(2, 0.5, num_of_blocks)
    base_body, base_shape = add_base(space, base_coord[0], base_coord[1], width=base_width)
    body_list = []
    shape_list = []
    body_list.append(base_body)
    shape_list.append(base_shape)
    for j in range(num_of_blocks):
        temp_block = base_body
        temp_shape = base_shape
        perm_block_dim = list(block_dim)
        if hor_ver_list[j]:
            perm_block_dim[0], perm_block_dim[1] = block_dim[1], block_dim[0]
        block_pos = [position_noise_list[j] * 100 + 300, 600]
        block_interval = interval([block_pos[0] - perm_block_dim[0] / 2., block_pos[0] + perm_block_dim[0] / 2.])
        for k in range(len(shape_list)):
            shape_k_interval = interval([shape_list[k].bb.left, shape_list[k].bb.right])
            if len(shape_k_interval & block_interval) > 0 and temp_shape.bb.top <= shape_list[k].bb.top:
                temp_block = body_list[k]
                temp_shape = shape_list[k]
        block_pos[1] = temp_shape.bb.top + perm_block_dim[1] / 2.
        body, shape = add_block(space, block_pos, mass=mass, block_dim=perm_block_dim)
        # time.sleep(1)
        body_list.append(body)
        shape_list.append(shape)
        # just for stopping the code from running
        if visual:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)

        # while body.velocity <=1:
        #     print body.force[0]
        if visual:
            draw_options = pygame_util.DrawOptions(screen)
            screen.fill((255, 255, 255))
            space.debug_draw(draw_options)
            clock.tick(50)
            pygame.display.flip()
        space.step(1 / 50.0)
        # for body in body_list:
        #     body._set_angular_velocity(0)
        #     body._set_velocity([0, 0])
        #     body._set_force([0, 0])
        #     body._set_torque(0)

        if visual:
            # just for stopping the code from running
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)

    return body_list, shape_list
                # body.body_type = pymunk.Body.STATIC


# def apply_noise_given_blocks(space, noise_trials):
#     num_of_blocks = len(space.bodies) - 1
#     for j in range(num_of_blocks):
#         space_copy = copy.deepcopy(space)
#         space_copy.gravity = (0.0, 0.0)
#         temp_array = np.zeros([noise_trials, num_of_blocks])
#         for k in range(noise_trials):
#             body_list = space_copy.bodies
#             shape_list = space_copy.shapes
#             moving_noise = truncnorm.rvs(-1. / 10., 1. / 10., size=1) * (
#                 shape_list[1].bb.right - shape_list[1].bb.left)
#             body_list[j].position = [body_list[j].position[0] + moving_noise, body_list[j].position[1]]
#             space_copy.reindex_shapes_for_body(body_list[j])
#             space_copy.gravity = (0.0, -900)
#             space_copy.step(1 / 50.)
#
#             for block in body_list:
#                 if block.angular_velocity > 0.01:
#                     temp_array[k, j] = 1.
#             for l in range(1, len(body_list)):
#                 space_copy.remove(body_list[l])
#                 space_copy.remove(shape_list[l])
#             space_copy.gravity = (0.0, 0.0)
#             if visual:
#                 screen.fill((255, 255, 255))
#                 # space.debug_draw(draw_options)
#                 clock.tick(50)
#                 pygame.display.flip()
#             space_copy.step(1 / 50.0)
#     return temp_array


def is_stable_func(position_var, num_of_blocks, noise_trials, space, arg_index):
    if space is None:
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
    position_noise_list = truncnorm.rvs(- 1. / (position_var), 1. / (position_var), size=num_of_blocks)
    hor_ver_list = np.random.binomial(2, 0.5, num_of_blocks)
    temp_array = np.zeros([noise_trials, num_of_blocks])
    is_unstable = dict()
    body_list, shape_list = smart_rain_maker(space, position_noise_list=position_noise_list,
                                             hor_ver_list=hor_ver_list,
                                             num_of_blocks=num_of_blocks, block_dim=[100, 40], var=1.,
                                             base_coord=[(0., 100.), (600., 100.)], base_width=10, mass=1)
    moving_noise = truncnorm.rvs(-1. / 20., 1. / 20., size=1) * (
        shape_list[1].bb.right - shape_list[1].bb.left)
    old_positions = [block.position for block in body_list]
    space.gravity = (0.0, -900.)
    if visual:
        draw_options = pygame_util.DrawOptions(screen)
    for lk in range(100):
        if visual:
            space.debug_draw(draw_options)
            clock.tick(50)
            pygame.display.flip()
        space.step(1 / 50.)
        if visual:
            screen.fill((255, 255, 255))
    for b_ind in range(1, len(body_list)):
        if abs(body_list[b_ind].position[1] - old_positions[b_ind][1]) > 10:
            # print block.angular_velocity
            is_unstable[tuple(body_list[b_ind].position)] = 1.
        else:
            is_unstable[tuple(body_list[b_ind].position)] = 0.
        if visual:
            screen.fill((255, 255, 255))
            # space.debug_draw(draw_options)
            clock.tick(50)
            pygame.display.flip()
        space.step(1 / 50.0)
    # return temp_array
    # the following line is temporary just for returning stablity
    return is_unstable


def apply_noise_part_B(position_var, num_of_blocks, noise_trials, space, arg_index):
    """ Parallel fucntion to be called"""
    if space is None:
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
    position_noise_list = truncnorm.rvs(- 1. / (position_var), 1. / (position_var), size=num_of_blocks)
    hor_ver_list = np.random.binomial(2, 0.5, num_of_blocks)
    temp_array = np.zeros([noise_trials, num_of_blocks])
    is_unstable = np.zeros(num_of_blocks)
    for j in range(num_of_blocks):
        for k in range(noise_trials):
            body_list, shape_list = smart_rain_maker(space, position_noise_list=position_noise_list,
                                                     hor_ver_list=hor_ver_list,
                                                     num_of_blocks=num_of_blocks, block_dim=[100, 40], var=1.,
                                                     base_coord=[(0., 100.), (600., 100.)], base_width=10, mass=1)
            moving_noise = truncnorm.rvs(-1. / 20., 1. / 20., size=1) * (
                shape_list[1].bb.right - shape_list[1].bb.left)

            # body_list[j].position = [body_list[j].position[0] + moving_noise , body_list[j].position[1]]
            space.reindex_shapes_for_body(body_list[j])
            body_left = shape_list[j].bb.left
            body_right = shape_list[j].bb.right
            body_top = shape_list[j].bb.top
            body_bottom = shape_list[j].bb.bottom

            # print 'length ',len(shape_list)
            # for counter in range(len(shape_list)):
            #     print shape_list[counter].bb.right - shape_list[counter].bb.left
            # real_BB = pymunk.BB([body_left, body_bottom, body_right, body_top])
            # while j>0 and (real_BB.intersects_segment((body_left, body_bottom), (body_right, body_top)) or \
            #         shape_list[j].bb.intersects_segment((body_left, body_top), (body_right, body_bottom))):
            #     body_list[j].position = [body_list[j].position[0] - moving_noise, body_list[j].position[1]]
            #     space.reindex_shapes_for_body(body_list[j])
            #     moving_noise = truncnorm.rvs(-1. / 10., 1. / 10., size=1) * (
            #         shape_list[1].bb.right - shape_list[1].bb.left)
            #     body_list[j].position = [body_list[j].position[0] + moving_noise, body_list[j].position[1]]
            #     space.reindex_shapes_for_body(body_list[j])

            old_positions = [block.position for block in body_list]
            space.gravity = (0.0, -900.)
            for i in range(50):
                space.step(1/50.)

            if visual:
                draw_options = pygame_util.DrawOptions(screen)
                for lk in range(100):
                    space.debug_draw(draw_options)
                    clock.tick(50)
                    pygame.display.flip()
                    space.step(1/50.)
                    screen.fill((255, 255, 255))

            for b_ind in range(len(body_list)):
                if abs(body_list[b_ind].position[1] - old_positions[b_ind][1]) > 5:
                    # print block.angular_velocity
                    temp_array[k, j] = 1.
                    is_unstable[j] = 1.
            for l in range(1, len(body_list)):
                space.remove(body_list[l])
                space.remove(shape_list[l])
            space.gravity = (0.0, 0.0)
            if visual:
                screen.fill((255, 255, 255))
                # space.debug_draw(draw_options)
                clock.tick(50)
                pygame.display.flip()
            space.step(1 / 50.0)
    # return temp_array
    # the following line is temporary just for returning stablity
    return is_unstable

def apply_noise_part_B_star(position_var, num_of_blocks, noise_trials, space, arg_index):
    return apply_noise_part_B(position_var, num_of_blocks, noise_trials, space, arg_index)

def apply_noise(block_arrangements_num=50000, noise_trials = 1, num_of_blocks = 9, noise_var = 1., position_var=1, space=None):
    success_array = np.zeros([block_arrangements_num, noise_trials, num_of_blocks])
    success_list = []
    if space is None:
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
    if parallel:
        pool = multiprocessing.Pool(processes=20)
        space = None
        func = functools.partial(apply_noise_part_B_star, position_var, num_of_blocks, noise_trials, space)
        m4 = pool.map(func, range(block_arrangements_num))
        output_list = map(lambda x: x, m4)
        for i in range(block_arrangements_num):
            success_array[i] = output_list[i]
    else:
        for i in range(block_arrangements_num):
            # the following two lines is temporary just for generating stable/unstable piles
            space = pymunk.Space()
            space.gravity = (0.0, 0.0)
            # success_array[i] = apply_noise_part_B(position_var, num_of_blocks, noise_trials, space, i)
            success_list.append(is_stable_func(position_var, num_of_blocks, noise_trials, space, i))

    return success_list
data_list = apply_noise()
f = open('/Users/naji/naji.pickle', 'w')
pickle.dump(data_list, f)




