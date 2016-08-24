import pymunk
from scipy.stats import truncnorm
from random import shuffle
import numpy as np
def add_block(space, position = [0.0,0.0], mass = 1, block_dim = [50, 50]):
    width = block_dim[1]
    height = block_dim[0]


    coords = [[- width / 2., - height / 2.], [-width /2. , height / 2.], [width / 2., height / 2.], [width / 2., - height / 2.]]
    moment = pymunk.moment_for_poly(mass, coords)
    body = pymunk.Body(mass,moment)
    body.position = position
    shape = pymunk.Poly(body, coords)
    space.add(body,shape)
    shape.friction = 1000000.0
    shape.elasticity = 0.0
    return body, shape


def make_pile(space, num_of_blocks = 5, base_coord = [(0., 100.), (500., 100.)], base_width = 10,  mass = 1, block_dim = [40, 100]):
    add_base(space, base_coord[0], base_coord[1], width=base_width)
    first_block_pos = [(base_coord[0][0] + base_coord[1][0]) / 2., base_coord[0][1] + base_width + block_dim[0] / 2.]
    shape_list = []
    body_list = []
    last_block_pos = first_block_pos
    for i in range(num_of_blocks):
        shuffle(block_dim)
        body, shape = add_block(space, last_block_pos , mass = mass, block_dim = block_dim)
        trunc_sample = truncnorm.rvs(- block_dim[0] / 20., block_dim[0] / 20., size = 1)[0] * 10 + last_block_pos[0]
        last_block_pos = [trunc_sample , last_block_pos[1] + block_dim[0]]
        body_list.append(body)
        shape_list.append(shape)
    return body_list, shape_list

def add_base(space, p1, p2, width = 10):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    mid =  p1[0], p1[1]
    body.position = mid
    shape = pymunk.Segment(body, (0,0), (p2[0]-p1[0],p2[1]-p1[1]), width)
    space.add(shape)
    shape.friction = 1000000.0
    shape.elasticity = 0.0
    return shape

