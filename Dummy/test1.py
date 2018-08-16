import numpy as np
import math
#from scipy.signal import resample
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# x = np.linspace(0, 10, num=10, endpoint=False)
# print (x)

# #fx = resample(x, 10)
# #for ff in fx:
# #    print (str(ff))

# #fx = resize(x, output_shape=[20])
# #print (fx)

# print (x.shape[0])

# def nn_interpolate_1d(A, new_size):
#     """Vectorized Nearest Neighbor Interpolation"""

#     old_size = A.shape
#     col_ratio = new_size/old_size[0]

#     col_idx = (np.ceil(range(1, 1 + int(old_size[0]*col_ratio))/col_ratio) - 1).astype(int)

#     final_matrix = A[col_idx, :]

#     return final_matrix

# #fx = nn_interpolate_1d(x, 20)
# #print (fx)

# #print (range(1, 5))

# a = [
#         [1,2,3]
#     ]
# b = [
#         [1,1,1],
#         [1,3,2],
#         [1,2,2]
#     ]

# similarity = cosine_similarity(a, b)
# print (similarity)

# print (np.argsort(similarity))

# print (np.fliplr(np.argsort(similarity))[0])

# sample_size = 10
# index = 6
# popedIndex = [i for i in range(0, sample_size)]

# print (popedIndex)

# popedIndex.pop(index)

# print (popedIndex)

a = np.array([1,4,73,24,972, 863, 92, 84, 739])
b = []

original_length = a.shape[0]
print (original_length)

# how long do you want it to be?
size = 30

# how many partition?
partition = math.floor(size/original_length)

print (partition)

repetition = 0
for i in range(1, size+1):
    b.append(a[repetition])
    if i%partition == 0 and i and repetition < original_length-1:
        repetition = repetition+1

print (b)

b = np.asarray(b)
c = b.reshape((3,5,2))

print (c)

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

print (stemmer.stem('tidak'))