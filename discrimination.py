import pymunk #1
import bisect
from scipy.stats import norm 
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from shapes import sort_pile, copy_space

def combined_center_of_mass(blocks, n_above = 1000, recog_noise = 1.0):
    nblocks = len(blocks)-1
    xcoms = [0.0]*(nblocks)
    labels = np.zeros(nblocks) 
    labels_det = np.zeros(nblocks)
    plane_heights = map(lambda block: block.bb.top,blocks[:-1])
    plane_lengths = map(lambda i: min(blocks[i].bb.right,blocks[i+1].bb.right) - \
                                  max(blocks[i].bb.left, blocks[i+1].bb.left), range(nblocks))
    plane_mids = map(lambda i: max(blocks[i].bb.left,blocks[i+1].bb.left) + plane_lengths[i]/2., range(nblocks))
    for bi in range(nblocks):
        upper_blocks = blocks[bi+1:min(bi+1+n_above, nblocks+1)]
        xcom = sum(map(lambda block:
            (block.body.position[0]-plane_mids[bi])*block.body.mass,
            upper_blocks)) # just consider one block above
        xcom /= sum(map(lambda block: block.body.mass, upper_blocks))
        labels_det[bi] = -1 if xcom < -0.5*plane_lengths[bi] else \
                +1 if xcom > 0.5*plane_lengths[bi] else 0
        labels[bi] = norm.cdf(0.5 * plane_lengths[bi], xcom, recog_noise) - \
                     norm.cdf(-0.5 * plane_lengths[bi], xcom, recog_noise)

    return plane_heights, labels, labels_det

def reset_space(space, pos):
    blocks = sort_pile(space.shapes)
    for bi, b in enumerate(blocks[1:]):
        b.body._set_position(tuple(pos[bi]))
        b.body._set_angle(0.0)
        b.body._set_velocity([0.0,0.0])
        b.body._set_angular_velocity(0.0)
        b.body._set_force([0.0,0.0])
        space.reindex_shapes_for_body(b.body)
    return space

def simulate_whole(space, recog_noise = 1.0, noise_rep = 30, det = False):
     
    blocks = sort_pile(space.shapes)
    nblocks = len(blocks)
    
    pos = np.array([bb.body.position.int_tuple for bb in blocks[1:]])
    pos_copy = deepcopy(pos)
    space_copy,_ = copy_space(space)
    py = pos[:,1]
    if not det: 
        results = np.zeros((noise_rep*(nblocks-1), nblocks-1))
        randx = np.random.randn(nblocks-1, noise_rep) * recog_noise
    else: 
        recog_noise = 0.0
        randx = np.zeros((1,1))
        results = np.zeros((1, nblocks-1))
        nblocks = 2
        noise_rep = 1
    count = 0
    for bi in range(nblocks-1):
        for ni in range(noise_rep):
            # reset_space(space, pos_copy)
            space,_ = copy_space(space_copy)
            blocks = sort_pile(space.shapes)
            pos = np.array([bb.body.position.int_tuple for bb in blocks[1:]])
            b = blocks[bi+1]
            px = b.body.position
            b.body.position = [px[0]+randx[bi,ni], px[1]]
            space.reindex_shapes_for_body(b.body)
            space.gravity = [0.0,-900.0]
            for ti in range(100):
                space.step(1/50.0)
            new_pys = np.array([bb.body.position.int_tuple[1] for bb in blocks[1:]])
            dy = py - new_pys
            results[count] = abs(dy)<30
            count+=1
    fall = results.all(1)
    results = results.mean(0)
    fall = fall.mean()

    pos = [(p[0],p[1]) for p in pos_copy]
    results = OrderedDict(zip(pos, results))
    reset_space(space, pos_copy)
    return results, fall
            

def is_pile(blocks):
    for bi in xrange(len(blocks)-1):
        if (blocks[bi+1].bb.bottom - blocks[bi].bb.top) != 0.0:
            return False
    return True


# deterministic level labels to block labels
def det_labels_to_block_labels(blocks, plane_heights, level_labels):
    num_blocks = len(blocks)-1
    unstable_level = np.nonzero(level_labels)[0]
    block_labels = OrderedDict(zip([(s.body.position.int_tuple)\
        for s in blocks[1:]], [True]*num_blocks))
    if len(unstable_level) == 0:
        return block_labels
    unstable_level = unstable_level[0]
    for k in block_labels.keys():
        block_labels[k] = k[1]<=plane_heights[unstable_level]

    return block_labels

# probablistic level labels to block labels
def labels_to_block_labels(blocks, plane_heights, level_labels):
    num_blocks = len(blocks)-1
    prob = np.ones(num_blocks)
    for i in range(num_blocks):
        prob[i:] *= level_labels[i] 
    
    block_labels = OrderedDict(zip([(s.body.position.int_tuple)\
        for s in blocks[1:]], prob))
    return block_labels
    
    
def labels_to_level_strips(dims, plane_heights, labels, n):
    
    lims = range(dims[0], dims[1]+1, (dims[1]-dims[0])/n)
    idx = map(lambda h: bisect.bisect_left(lims, h), plane_heights)
    vec = [0]*n
    for ii, i in enumerate(idx):
        vec[i] = labels[ii]
    return vec


# ============
# = Using simulation
# ============

