import pymunk
from random import shuffle
import numpy as np
from scipy.stats import truncnorm
import inspect
from interval import interval
import re

def add_block(space, position = [0.0,0.0], mass = 1, block_dim = [50, 50], radius=1):
    width = block_dim[0]
    height = block_dim[1]

    coords = [[- width / 2.+radius, - height / 2.+radius], [-width /2.+radius , height / 2.-radius], [width / 2.-radius, height / 2.-radius], [width / 2.-radius, - height / 2.+radius]]
    moment = pymunk.moment_for_poly(mass, coords)
    body = pymunk.Body(mass,moment)
    body.position = position
    shape = pymunk.Poly(body,coords, radius = radius)
    space.add(body,shape)
    shape.friction = 1
    shape.elasticity = 0.0
    return body, shape

def make_pile_given_noise(space=None, base_coord = [(0., 100.), (600., 100.)], base_width = 10,  mass = 1,
                          block_dim = [100,40], position_noise_list=[0] * 5, hor_ver_list = [0] * 5):
    if space is None:
        space = []
    num_of_blocks = len(position_noise_list)
    # making the base
    add_base(space, base_coord[0], base_coord[1], width=base_width)

    shape_list = []
    body_list = []

    #copying block_dim list
    perm_block_dim = block_dim[:]
    # swapping the coordinates if hor_ver_list is set to 1
    # perm_block_dim is the same as block_dim due to the = in line 31, so if you swap perm_block_dim, it also swaps block_dim...
    if hor_ver_list[0]:
        perm_block_dim[0], perm_block_dim[1] = block_dim[1], block_dim[0]
    first_block_pos = [(base_coord[0][0] + base_coord[1][0]) / 2., base_coord[0][1] + base_width + perm_block_dim[1] / 2.]


    last_block_pos = [[], []]
    last_block_pos = first_block_pos[:]
    last_top = last_block_pos[1] + perm_block_dim[1] / 2.
    last_block_width = perm_block_dim[0]
    for i in range(num_of_blocks):
        body, shape = add_block(space, last_block_pos, mass = mass, block_dim = perm_block_dim)
        perm_block_dim = block_dim[:]
        # swapping the coordinates if hor_ver_list is set to 1
        if hor_ver_list[i]:
            perm_block_dim[0], perm_block_dim[1] = block_dim[1], block_dim[0]


        x_range = last_block_width / 2. + perm_block_dim[0] / 2.
        trunc_sample = position_noise_list[i] * x_range + last_block_pos[0]
        last_block_pos = [trunc_sample , last_top + perm_block_dim[1] / 2.]
        last_top = last_block_pos[1] + perm_block_dim[1] / 2.
        last_block_width = perm_block_dim[0]
        body_list.append(body)
        shape_list.append(shape)
    print space.shapes
    return body_list, shape_list




def make_pile(space, num_of_blocks = 5, base_coord = [(0., 100.), (500., 100.)], \
        base_width = 10,  mass = 1, block_dim = [100,40], noise=1, tough = False):

    _, base_shape = add_base(space, base_coord[0], base_coord[1], width=base_width)
    first_block_pos = [(base_coord[0][0] + base_coord[1][0]) / 2., base_coord[0][1] + base_width + block_dim[1] / 2.]
    shape_list = []
    last_block_pos = first_block_pos
    last_top = last_block_pos[1] + block_dim[1] / 2.
    trunc_sample = truncnorm.rvs(- 1. / (noise), 1. / (noise), size = num_of_blocks) * noise
    shape_list.append(base_shape)
    for i in range(num_of_blocks):
        body, shape = add_block(space, last_block_pos, mass = mass, block_dim = block_dim)
        last_block_width = block_dim[0]
        shuffle(block_dim)
        x_range = last_block_width/2 # + block_dim[0]/2
        if not tough:
            x_range += block_dim[0]/4
        x_pos = trunc_sample[i] * x_range + last_block_pos[0]
        last_block_pos = [x_pos, last_top + block_dim[1]/2.]
        last_top = last_block_pos[1] + block_dim[1]/2.
        shape_list.append(shape)
    sort_pile(shape_list) 
    return shape_list

def copy_space(space):

    shapes = sort_pile(space.shapes)
    bodies = [s.body for s in shapes]
    n = len(shapes)
    new_space = pymunk.Space()
    shape_list = []
    base_shape = shapes[0]
    base_coords = [(base_shape.bb.left, base_shape.body.position[1]),\
                    (base_shape.bb.right, base_shape.body.position[1])]
    base_width = base_shape.bb.top - base_shape.body.position[1]
    _, base_shape = add_base(new_space, base_coords[0], base_coords[1], width = base_width) 
    shape_list.append(base_shape)
    for i in range(n-1):
        block = shapes[i+1]
        block_center = block.body.position
        block_mass = block.body.mass
        block_dim = [(block.bb.right - block.bb.left), (block.bb.top-block.bb.bottom)]
        _, shape = add_block(new_space, block_center, mass = block_mass, block_dim = block_dim)
        shape_list.append(shape)
    return new_space, shape_list
    
def add_base(space, p1, p2, width = 5, radius = 1):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    mid =  (p1[0] + p2[0])/2.0, (p1[1] + p2[1])/2.0
    half_length = (p2[0]-p1[0])/2.0
    body.position = mid
    coords = [[-half_length+radius,-width+radius], [+half_length-radius,-width+radius], \
        [-half_length+radius, +width-radius], [+half_length-radius, +width-radius]]
    shape = pymunk.Poly(body, coords, radius = radius)
    space.add(shape)
    shape.friction = 1
    shape.elasticity = 0.0
    return body, shape

def sort_pile(blocks):
    blocks.sort(key = lambda x: (x.body.position[1], x.body.position[0]), reverse = False)
    return blocks


def smart_rain_maker(space, num_of_blocks=5, block_dim=[100, 40], var=1., base_coord=[(0., 100.), (600., 100.)], base_width=10, mass=1.0):
    position_noise_list = truncnorm.rvs(- 1. / (var), 1. / (var), size=num_of_blocks)
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
        block_pos = [position_noise_list[j] * block_dim[0] + (base_coord[1][0]/2), 1000]
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

    return body_list, shape_list
