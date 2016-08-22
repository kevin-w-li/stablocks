import pymunk
from scipy.stats import truncnorm
def add_block(space, position = [0.0,0.0], mass = 1, size = 10):

    coord = size/2
    coords = [[-coord, -coord], [coord,-coord], [coord, coord], [-coord,coord]]
    moment = pymunk.moment_for_poly(mass, coords)
    body = pymunk.Body(mass,moment)
    body.position = position
    shape = pymunk.Poly(body, coords)
    space.add(body,shape)
    shape.friction = 1000000.0
    shape.elasticity = 0.0
    return body, shape


def make_pile(space, num_of_blocks = 5, base_coord = [(0., 100.), (500., 100.)], base_width = 10,  mass = 1, size = 50):
    add_base(space, base_coord[0], base_coord[1], width=base_width)
    first_block_pos = [(base_coord[0][0] + base_coord[1][0]) / 2., base_coord[0][1] + base_width + size / 2.]
    print first_block_pos
    shape_list = []
    body_list = []
    last_block_pos = first_block_pos
    for i in range(num_of_blocks):
        body, shape = add_block(space, last_block_pos , mass = mass, size = size)


        trunc_sample = truncnorm.rvs(- size / .002, size / .002, size = 1)[0] + last_block_pos[0]
        last_block_pos = [trunc_sample , last_block_pos[1] + size / 2.]
        print ":salam",last_block_pos[0] - size / 2., last_block_pos[0] + size /2. ,"khszk"
        print last_block_pos
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

