import sys
import pygame
from pygame.locals import *
import pymunk #1
import random
from shapes import add_block, add_base, make_pile



pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Blocks will fall?")
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0.0, -900.0)

blocks = []
from pymunk import pygame_util
draw_options = pygame_util.DrawOptions(screen)

# add the base block, need to figure out how to set positoin...
base = add_base(space, (0,100),(500,100))

ticks_to_next_block = 10
s = make_pile(space, 6)

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


    space.step(1/50.0)

    screen.fill((255,255,255))

    space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(50)





