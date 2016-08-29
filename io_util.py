from matplotlib import pyplot as plt
import matplotlib as mpl
import pymunk, io
from PIL import Image, ImageFilter
import numpy as np
import h5py
import cPickle as pkl
from shapes import sort_pile

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

def space_label_to_array(space,label, display_size, image_size, fig, ax, plt_options):
    space.debug_draw(plt_options)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    blocks = filter(lambda c: isinstance(c, mpl.patches.Polygon), ax.get_children())
    for p in blocks:
        center = tuple(np.mean(p.get_xy()[0:4],0).astype(int))
        if center not in label: continue # the base does not have lables
        p.set_facecolor(np.tile([label[center]], 3))
	p.set_edgecolor([1.0,1.0,1.0])
            
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

def plot_many_piles_slices(data,label,vec):
   
    assert data.ndim == 4 and vec.ndim == 2
    n = min(data.shape[0], 5)
    image_size = data.shape[1]
    label_size = label.shape[1]
    num_slices = vec.shape[1]

    fig,axes = plt.subplots(2,n,figsize = (n*4,5))
    for i in range(n):
        axes[0,i].imshow(data[i])
        axes[0,i].set(adjustable='box-forced', aspect=1, xlim=(0,image_size), ylim=(0,image_size))
        axes[0,i].invert_yaxis()
        ax = axes[1,i]
        ax.imshow(label[i])
        ax.set(adjustable='box-forced', aspect=1, xlim=(0,label_size), ylim=(0,label_size))
        ax2 = ax.twinx()
        ax2.set(adjustable='box-forced', aspect=1, xlim=(0,label_size), ylim=(0,label_size))
        ax.invert_yaxis()
        ax2.barh(label_size/float(num_slices)*np.linspace(0.5,num_slices-0.5,num_slices), vec[i]*label_size*0.2, align='center')
        #ax.set_axis_off()
        ax2.set_axis_off()
    plt.show()

def load_hdf5(filename):
   # load from hdf5
   f = h5py.File(filename, 'r')
   return np.array(f['data']), np.array(f['label'])
   
def save_space(spaces, labels, filename):
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
        pkl.dump(dict(data=data, label=labels), f)

def load_space(filename):

    with open(filename, 'r') as f:
        data_label = pkl.load(f)
        data = data_label['data']
        label = data_label['label']

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
    return spaces, label
