import sys, io
import pygame
from matplotlib import pyplot as plt
from pygame.locals import *
from pygame import draw
import pymunkoptions
pymunkoptions.options["debug"] = False
import pymunk #1
from pymunk import pygame_util
from pymunk import matplotlib_util
from shapes import *
from discrimination import *
import numpy as np
from io_util import *
from time import strftime, sleep
from collections import OrderedDict

display_size = (1880,1000)
# display_size = (1000,1000)
stable_color = [52,152,219]
unstable_color = [250,0,0]
pygame.init()
screen = pygame.display.set_mode(display_size)

dataset_name = "spaces50"
spaces,labels = load_space('exp/' + dataset_name)
num_piles = len(spaces)
responses= [None]*len(spaces)

def toggle_highlight(space,screen, mouse,  resp):
    mouse = mouse[0], display_size[1] - mouse[1]
    for s in space.shapes:
        # base should not be highlighted
        if s.bb.left < 0: continue
        if s.point_query(mouse):
            qinfo = s.point_query(mouse)
            if qinfo.distance < 0:
                vertices = s.get_vertices()
                for v in vertices:
                    if v[0] < 0:
                        v[0] += 2
                    else:
                        v[0] -= 1

                    if v[1] < 0:
                        v[1] += 2
                    else:
                        v[1] -= 1
                pos = s.body.position.int_tuple
                vertices = [(v[0] + pos[0], display_size[1]-(v[1] + pos[1]-1)) for v in vertices]
                assert pos in resp
                if resp[pos]:
                    pygame.draw.polygon(screen, unstable_color, vertices,  0)
                else:
                    pygame.draw.polygon(screen, stable_color, vertices,  0)
                resp[pos] = not resp[pos]
                pygame.display.flip()

def add_highlight(space, screen, pos):
    draw_highlight(space,screen, pos, [250,0,0])

def remove_highlight(space, screen, pos):
    draw_highlight(space,screen, pos, [52,152,219])

def draw_blocks(screen, space, count):
    screen.fill((250,250,250))
    for shape in space.shapes:
        pos = shape.body.position
        shape.body.position = [pos[0] + (display_size[0]-1000)/2, pos[1]]
        space.reindex_shapes_for_body(shape.body)
    draw_options = pygame_util.DrawOptions(screen)
    color = draw_options.color_for_shape(space.shapes[1])
    color = [color.r, color.g, color.b, 0]
    color = [0,0,0, 0]
    draw_options.shape_outline_color = (0,0,0,255)
    draw_options.collision_point_color = color
    space.debug_draw(draw_options)
    display_text(count)

background = pygame.Surface([1000,200])
background = background.convert()

def display_text(count):
    fontsize = 26
    font = pygame.font.SysFont("Arial", fontsize)
    background.fill((250, 250, 250))
    texts = []
    texts.append(font.render("Select all blocks that will fall", 1, (10, 10, 10)))
    texts.append(font.render("Left mouse to toggle selection", 1, (10, 10, 10)))
    texts.append(font.render("Press Enter to confirm and move on", 1, (10, 10, 10)))
    texts.append(font.render("You may press Enter with no selection (stable tower)", 1, (10, 10, 10)))
    texts.append(font.render(str(count) + " out of " + str(num_piles) + ' done', 1, (10, 10, 10)))
    for i, text in enumerate(texts):
        textpos = texts[i].get_rect()
        textpos.top = 10+fontsize*i
        textpos.left = 10
        background.blit(texts[i], textpos)

    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pygame.display.flip()


def finish():
        
    np.save('exp/resp/'+strftime('%m-%d-%H-%M'), np.array(responses))

    fontsize = 26
    font = pygame.font.SysFont("Arial", fontsize)
    
    correct = sum(map(lambda i:\
        all(responses[i].values()) == all(labels[i].values()), range(num_piles)))

    texts = []
    texts.append(font.render("Thank you very much!", 1, (10, 10, 10)))
    texts.append(font.render("You got " + str(correct) + " correct!", 1, (10, 10, 10)))
    background.fill((250, 250, 250))
    for i, text in enumerate(texts):
        textpos = texts[i].get_rect()
        textpos.top = 10+fontsize*i
        textpos.left = 10
        background.blit(texts[i], textpos)

    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pygame.display.flip()
    sleep(5) 
    pygame.quit()
    quit()
    
count = 0
# Fill background

while True:
    event = pygame.event.wait()
    #for event in pygame.event.get():
    if pygame.mouse.get_pressed()[0]:
        if count == 0:
            continue
        pos = pygame.mouse.get_pos()
        toggle_highlight(space, screen, pos, responses[count-1])
        
    if pygame.key.get_pressed()[pygame.K_RETURN] != 0:
        while 1:
            event = pygame.event.wait()
            if pygame.key.get_pressed()[pygame.K_RETURN] == 0:
                break
        if count == num_piles:
            print 'last one'
            finish()

        space = spaces[count]
        draw_blocks(screen, space, count)
        blocks = sort_pile(space.shapes)
        num_blocks = len(blocks)
        ys = [s.body.position.int_tuple for s in blocks[1:]]
        pygame.display.flip()
        responses[count] = OrderedDict(zip(ys, [True]*num_blocks))
        count+=1
        pygame.time.wait(10)
