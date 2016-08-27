from io_util import *
filename = 'data/dataset_10_5_5.hdf5'
all_data, all_slices = load_hdf5(filename)
plot_many_piles_slices(all_data,all_slices)
