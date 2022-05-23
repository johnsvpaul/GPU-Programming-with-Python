import numpy as np
from imageio import imread, imsave
from time import time  # Import time tools

# Read the image
image = "1S"
noiseLevel = ["30NOISE", "50"]  # percent of noise

img_noisy1 = imread(
    "dataset/"+noiseLevel[0]+"/"+image+".png").astype(np.float32)

# Obtain the number of rows and columns
# of the image
m, n = img_noisy1.shape

# Traverse the image. For every 3X3 area,
# find the median of the pixels and
# replace the center pixel by the median
img_new1 = np.zeros([m, n])
cpu_start_time = time()  # Get the CPU start time
for i in range(1, m-1):
    for j in range(1, n-1):
        temp = [img_noisy1[i-1, j-1],
                img_noisy1[i-1, j],
                img_noisy1[i-1, j + 1],
                img_noisy1[i, j-1],
                img_noisy1[i, j],
                img_noisy1[i, j + 1],
                img_noisy1[i + 1, j-1],
                img_noisy1[i + 1, j],
                img_noisy1[i + 1, j + 1]]

        temp = sorted(temp)
        img_new1[i, j] = temp[4]
cpu_end_time = time()  # Get the CPU end time
print("CPU Time: {0} s".format(cpu_end_time - cpu_start_time))
img_new1 = img_new1.astype(np.uint8)
imsave("results/filtered"+noiseLevel[0]+image+".png", img_new1)
