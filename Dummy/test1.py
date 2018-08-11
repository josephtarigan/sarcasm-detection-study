import numpy as np
#from scipy.signal import resample
from skimage.transform import resize

x = np.linspace(0, 10, num=10, endpoint=False)
print (x)

#fx = resample(x, 10)
#for ff in fx:
#    print (str(ff))

#fx = resize(x, output_shape=[20])
#print (fx)

print (x.shape[0])

def nn_interpolate_1d(A, new_size):
    """Vectorized Nearest Neighbor Interpolation"""

    old_size = A.shape
    col_ratio = new_size/old_size[0]

    col_idx = (np.ceil(range(1, 1 + int(old_size[0]*col_ratio))/col_ratio) - 1).astype(int)

    final_matrix = A[col_idx, :]

    return final_matrix

#fx = nn_interpolate_1d(x, 20)
#print (fx)

print (range(1, 5))