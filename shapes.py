import pymunk

def add_block(space, position = [0.0,0.0], mass = 1, size = 10):

    coord = size/2
    coords = [[-coord, -coord], [coord,-coord], [coord, coord], [-coord,coord]]
    moment = pymunk.moment_for_poly(mass, coords)
    body = pymunk.Body(mass,moment)
    body.position = position
    shape = pymunk.Poly(body, coords)
    space.add(body,shape)
    return shape


def add_base(space, p1, p2):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    mid = [ (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 ]
    body.position = (mid[0],mid[1])
    l = pymunk.Segment(body, p1, p2,5)
    space.add(l)
    return l

