import pymunk
from scipy.stats import truncnorm, uniform
from random import shuffle
import numpy as np
def add_block(space, position = [0.0,0.0], mass = 1, block_dim = [50, 50]):
    width = block_dim[0]
    height = block_dim[1]

    coords = [[- width / 2., - height / 2.], [-width /2. , height / 2.], [width / 2., height / 2.], [width / 2., - height / 2.]]
    moment = pymunk.moment_for_poly(mass, coords)
    body = pymunk.Body(mass,moment)
    body.position = position
    shape = pymunk.Poly(body,coords)
    space.add(body,shape)
    shape.friction = 1000.0
    shape.elasticity = 0.0
    return body, shape


def make_pile(space, num_of_blocks = 5, base_coord = [(0., 100.), (500., 100.)], base_width = 10,  mass = 1, block_dim = [100,40], noise=1):

    add_base(space, base_coord[0], base_coord[1], width=base_width)
    first_block_pos = [(base_coord[0][0] + base_coord[1][0]) / 2., base_coord[0][1] + base_width + block_dim[1] / 2.]
    shape_list = []
    body_list = []
    last_block_pos = first_block_pos
    last_top = last_block_pos[1] + block_dim[1] / 2.
    trunc_sample = truncnorm.rvs(- 1. / (noise), 1. / (noise), size = num_of_blocks) * noise
    for i in range(num_of_blocks):
        body, shape = add_block(space, last_block_pos, mass = mass, block_dim = block_dim)
        last_block_width = block_dim[0]
        shuffle(block_dim)
        x_range = last_block_width/2 + block_dim[0]/2
        x_pos = trunc_sample[i] * x_range + last_block_pos[0]
        last_block_pos = [x_pos, last_top + block_dim[1]/2.]
        last_top = last_block_pos[1] + block_dim[1]/2.
        body_list.append(body)
        shape_list.append(shape)
    return body_list, shape_list
'''
def make_pile(space, num_of_blocks = 5, base_coord = [(0, 100), (500, 100)], mass = 1, size = 10):
    add_base(space, base_coord[0], base_coord[1])
    first_block_pos = [(base_coord[0][0] + base_coord[1][0]) / 2., base_coord[0][1]]
    shape_list = []
    body_list = []
    for i in range(num_of_blocks):
        body, shape = add_block(space, first_block_pos , mass = mass, size = size)
        body_list.append(body)
        shape_list.append(shape)
    return body_list, shape_list
'''
def add_base(space, p1, p2, width = 5):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    mid =  (p1[0] + p2[0])/2.0, (p1[1] + p2[1])/2.0
    half_length = (p2[0]-p1[0])/2.0
    body.position = mid
    coords = [[-half_length,-width], [+half_length,-width], \
        [-half_length, +width], [+half_length, +width]]
    shape = pymunk.Poly(body, coords)
    space.add(shape)
    shape.friction = 1000.0
    shape.elasticity = 0.0
    return shape

