import h5py

with h5py.File('20191029_0208_color.hdf5', 'r') as f:
    print(f.keys())
    a = f['20191029_0208']
    print(a['0'].value)
    print(a['0'].dtype)