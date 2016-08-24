from matplotlib import pyplot as plt
import pymunk, io
from PIL import Image
import numpy as np


def space_to_array(space, display_size, image_size):

    # space: pymunk space that contains shapes
    # display_size: display size that is used in space
    # image_size: size of the image data

    fig,ax = plt.subplots(1, figsize = (6,6))
    plt_options = pymunk.matplotlib_util.DrawOptions(ax)
    ax.set(adjustable='box-forced', aspect=1, xlim=(0,display_size), ylim=(0, display_size))
    space.debug_draw(plt_options)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=extent)
    buf.seek(0)
    im = Image.open(buf)
    im = im.resize((image_size,image_size),  Image.ANTIALIAS)
    data = np.array(im)
    data = data[:,:,0:3]
    return data
