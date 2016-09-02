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
from time import strftime, sleep, time
from collections import OrderedDict
import sys, io, pyautogui
import vidcap, subprocess

pygame.init()
display_size = (1920,1000)
stable_color = [52,152,219]
unstable_color = [250,0,0]
screen = pygame.display.set_mode(display_size, HWSURFACE | DOUBLEBUF | RESIZABLE)

dataset_names = [
"exp_20_4_3_space", 
"exp_50_5_3_space",
"exp_50_7_3_space",
"exp_multi_30_space"
]
'''
dataset_names = [
"exp_multi_30_space"
]
'''
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
                if resp['choices'][pos]:
                    pygame.draw.polygon(screen, unstable_color, vertices,  0)
                else:
                    pygame.draw.polygon(screen, stable_color, vertices,  0)
                resp['choices'][pos] = not resp['choices'][pos]
                resp['seq'].append((time(), pos))
                print resp
                pygame.display.flip()
    return resp

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
    color = [0,0,0,0]
    draw_options.shape_outline_color = (0,0,0,255)
    draw_options.collision_point_color = color
    space.debug_draw(draw_options)
    display_text(count)

background = pygame.Surface([800,200])
background = background.convert()

def display_text(count):
    fontsize = 26
    font = pygame.font.SysFont("Arial", fontsize)
    background.fill((250, 250, 250))
    texts = []
    texts.append(font.render("Select all blocks that will fall", 1, (10, 10, 10)))
    texts.append(font.render("Left mouse to toggle selection", 1, (10, 10, 10)))
    texts.append(font.render("Blue (default) means stable, red means unstable", 1, (10, 10, 10)))
    texts.append(font.render("Press Enter to confirm and move on to next tower", 1, (10, 10, 10)))
    texts.append(font.render("You may press Enter with no selection (stable tower)", 1, (10, 10, 10)))
    texts.append(font.render(str(count) + " out of " + str(num_piles) + ' done', 1, (10, 10, 10)))
    for i, text in enumerate(texts):
        textpos = texts[i].get_rect()
        textpos.top = 20+fontsize*i
        textpos.left = 50
        background.blit(texts[i], textpos)

    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pygame.display.flip()


def finish(lastone, responses):
        

    fontsize = 26
    font = pygame.font.SysFont("Arial", fontsize)
    correct = sum(map(lambda i:\
        all(responses[i]['choices'].values()) == all(labels[i].values()), range(num_piles)))

    texts = []
    if lastone:
        texts.append(font.render("Thank you very much!", 1, (10, 10, 10)))
    else: 
        texts.append(font.render("End of this set, take a rest", 1, (10, 10, 10)))
    texts.append(font.render("You got " + str(correct) + ' out of ' +str(num_piles) +" correct!", 1, (10, 10, 10)))
    background.fill((250, 250, 250))
    for i, text in enumerate(texts):
        textpos = texts[i].get_rect()
        textpos.top = 20+fontsize*i
        textpos.left = 50
        background.blit(texts[i], textpos)

    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pygame.display.flip()
    
# Fill background

all_responses= OrderedDict(zip(dataset_names,[None]*len(dataset_names)))
for dataset_name in dataset_names:

    spaces,labels = load_space('exp/' + dataset_name)
    num_piles = len(spaces)
    responses = []
    for i in range(num_piles):
        responses.append(dict(choices=None, seq=None))

    count = 0

    while True:
        event = pygame.event.wait()
        # for event in pygame.event.get():
        if pygame.mouse.get_pressed()[0]:
            while 1:
                event = pygame.event.wait()
                if pygame.mouse.get_pressed()[0] == 0:
                    break
            if count == 0:
                continue
            pos = pygame.mouse.get_pos()
            toggle_highlight(space, screen, pos, responses[count-1])
            pygame.time.wait(5)
            
        if pygame.key.get_pressed()[pygame.K_RETURN] != 0:
            while 1:
                event = pygame.event.wait()
                if pygame.key.get_pressed()[pygame.K_RETURN] == 0:
                    break
            if count == num_piles:
                all_responses[dataset_name] = responses
                finish(dataset_name == dataset_names[-1], responses)
                sleep(3) 
                if dataset_name == dataset_names[-1]:
                    pkl.dump(all_responses, open('exp/resp/'+strftime('%m-%d-%H-%M'),'w'))
                    pygame.quit()
                    # subprocess.Popen('python vidcap.py')
                    quit()
                else: break
            space = spaces[count]
            draw_blocks(screen, space, count)
            blocks = sort_pile(space.shapes)
            num_blocks = len(blocks)
            ys = [s.body.position.int_tuple for s in blocks[1:]]
            pygame.display.flip()
            responses[count]['choices'] = OrderedDict(zip(ys, [True]*num_blocks))
            responses[count]['seq'] = [(time(), (0,0))]
            count+=1
            pyautogui.moveTo(700,1080/2)
            pygame.time.wait(10)

