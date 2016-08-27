from matplotlib import pyplot as plt
import pymunk, io
from PIL import Image
import numpy as np
import h5py


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
   print f.keys()
   return np.array(f['data']), np.array(f['label'])
   
