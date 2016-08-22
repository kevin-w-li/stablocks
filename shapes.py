import pymunk

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
    return shape


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

def add_base(space, p1, p2):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    mid =  p1[0], p1[1]
    body.position = mid
    shape = pymunk.Segment(body, (0,0), (p2[0]-p1[0],p2[1]-p1[1]), 10)
    space.add(shape)
    shape.friction = 1000000.0
    shape.elasticity = 0.0
    return shape

