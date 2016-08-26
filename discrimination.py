import pymunk #1
import bisect
from scipy.stats import norm 
import numpy as np

def sort_pile(blocks):
    assert is_pile(blocks)
    blocks.sort(key = lambda x: x.body.position, reverse = False)
    return blocks

def combined_center_of_mass(blocks, recog_noise = 1.0):

    nblocks = len(blocks)
    xcoms = [0.0]*(nblocks-1)
    labels = [0]*(nblocks-1) 
    plane_heights = map(lambda block: block.bb.top,blocks[:-1])
    plane_lengths = map(lambda i: min(blocks[i].bb.right,blocks[i+1].bb.right) - \
                                  max(blocks[i].bb.left, blocks[i+1].bb.left), range(nblocks-1))
    plane_mids = map(lambda i: max(blocks[i].bb.left,blocks[i+1].bb.left) + plane_lengths[i]/2., range(nblocks-1))
    print np.array(plane_lengths)*227/600.0
    print np.array(plane_mids) * 227/600.0
    for bi, b in enumerate(blocks[:-1]):
        xcom = sum(map(lambda block:
            (block.body.position[0]-plane_mids[bi])*block.body.mass,
            blocks[bi+1:]))
        xcom /= sum(map(lambda block: block.body.mass, blocks[bi+1:]))
        if recog_noise == 0.0:
            labels[bi] = -1 if xcom < -0.5*plane_lengths[bi] else \
                    +1 if xcom > 0.5*plane_lengths[bi] else 0
        else:
            labels[bi] = norm.cdf(0.5 * plane_lengths[bi], xcom, recog_noise) - \
                         norm.cdf(-0.5 * plane_lengths[bi], xcom, recog_noise)
        print labels[bi]
    return plane_heights, labels

def is_pile(blocks):

    for bi in xrange(len(blocks)-1):
        if (blocks[bi+1].bb.bottom - blocks[bi].bb.top) != 0.0:
            return False
    
    return True
    
def labels_to_strips(dims, plane_heights, labels, n):
    
    lims = range(dims[0], dims[1]+1, (dims[1]-dims[0])/n)
    idx = map(lambda h: bisect.bisect_left(lims, h), plane_heights)
    vec = [0]*n
    for ii, i in enumerate(idx):
        vec[i] = labels[ii]
    return vec
