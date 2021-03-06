""" Code generating random piles of blocks
"""
import sys
import pymunk #1
import random
from shapes import add_block, add_base, make_pile


def generate_piles(trials = 10):
    space = pymunk.Space()
    space.gravity = (0.0, -900.0)

    return_list = []
    # the loop making difference spaces
    for i in range(trials):
        space = pymunk.Space()
        space.gravity = (0.0, -900.0)
        base = add_base(space, (0,100),(500,100))
        _, s = make_pile(space, 3)
        return_list.append(s)
    return s

if __name__ ==  "__main__":
    print generate_piles()
