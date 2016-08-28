from matplotlib import pyplot as plt
import pymunk, io
from PIL import Image, ImageFilter
import numpy as np
import h5py
from discrimination import sort_pile
import cPickle as pkl


def space_to_array(space, display_size, image_size, fig, ax, plt_options):

    # space: pymunk space that contains shapes
    # display_size: display size that is used in space
    # image_size: size of the image data

    space.debug_draw(plt_options)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=extent)
    buf.seek(0)
    im = Image.open(buf)
    im = im.resize((image_size,image_size),  Image.ANTIALIAS)
    # im = im.filter( ImageFilter.SHARPEN )
    data = np.array(im)
    data = data[:,:,0:3]
    plt.close()
    data = np.array(data, dtype = np.uint8)
    return data

def plot_pile_slice(data, vec):

    image_size = data.shape[0]
    num_slices = len(vec)
    fig,ax = plt.subplots()
    ax.imshow(data)
    ax.set(adjustable='box-forced', aspect=1, xlim=(0,image_size), ylim=(0,image_size))
    ax2 = ax.twinx()
    ax2.set(adjustable='box-forced', aspect=1, xlim=(0,image_size), ylim=(0,image_size))
    ax.set_ylim([0,image_size])
    ax.invert_yaxis()
    ax2.barh(image_size/float(num_slices)*np.linspace(0.5,num_slices-0.5,num_slices), vec*20, align='center')
    ax.set_axis_off()
    ax2.set_axis_off()
    plt.show()

def plot_many_piles_slices(data,vec):
   
    assert data.ndim == 4 and vec.ndim == 2
    n = min(data.shape[0], 5)
    image_size = data.shape[1]
    num_slices = vec.shape[1]

    fig,axes = plt.subplots(1,n,figsize = (n*4,5))
    for i, ax in enumerate(axes):
        ax.imshow(data[i])
        ax.set(adjustable='box-forced', aspect=1, xlim=(0,image_size), ylim=(0,image_size))
        ax2 = ax.twinx()
        ax2.set(adjustable='box-forced', aspect=1, xlim=(0,image_size), ylim=(0,image_size))
        ax.set_ylim([0,image_size])
        ax.invert_yaxis()
        ax2.barh(image_size/float(num_slices)*np.linspace(0.5,num_slices-0.5,num_slices), vec[i]*20, align='center')
        #ax.set_axis_off()
        ax2.set_axis_off()
    plt.show()

def load_hdf5(filename):
   # load from hdf5
   f = h5py.File(filename, 'r')
   return np.array(f['data']), np.array(f['label'])
   
def save_space(spaces, filename):
    num_blocks = len(spaces)
    data = [None]*num_blocks
    for si, space in enumerate(spaces):
        props  = dict(gravity = space.gravity, iterations = space.iterations)
        blocks = [None]*len(space.shapes)
        pile = sort_pile(space.shapes)
        for i, s in enumerate(pile):
            pos = s.body.position
            mass = s.body.mass
            moment = s.body.moment
            vert = s.get_vertices()
            fric = s.friction
            elas = s.elasticity
            blocks[i] = dict(pos = pos, mass = mass, moment= moment, \
                vert = vert, fric = fric, elas = elas)
        datum = dict(blocks = blocks, props = props)
        data[si] = datum

    with open(filename,'w') as f:
        pkl.dump(data, f)

def load_space(filename):

    with open(filename, 'r') as f:
        data = pkl.load(f)
    num_spaces = len(data)
    spaces = [pymunk.Space()]*num_spaces
    for di, datum in enumerate(data):
        spaces[di] = pymunk.Space()
        for k, v in datum['props'].iteritems():
            spaces[di].__setattr__(k,v)
        base_data = datum['blocks'][0]
        base_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        base_body.position = base_data['pos']
        shape = pymunk.Poly(base_body, base_data['vert'], radius = 1)
        spaces[di].add(shape)
        for i, d in enumerate(datum['blocks'][1:]):
            body = pymunk.Body(d['mass'], d['moment'])
            body.position = d['pos']
            shape = pymunk.Poly(body, d['vert'], radius = 1)
            shape.friction = d['fric']
            shape.elasticity = d['elas']
            spaces[di].add(body,shape)
    return spaces
