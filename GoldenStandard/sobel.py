from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from time import time  # Import time tools
from imageio import imread, imsave
# importing image
image_file = 'dataset/1S.png'
# this is the array representation of the input image
input_image = imread(image_file)

# #---------------------------------------------------------------------------------------------------------------------
#  Applying the Sobel operator
# #---------------------------------------------------------------------------------------------------------------------

"""
The kernels Gx and Gy can be thought of as a differential operation in the "input_image" array in the directions x and y 
respectively. These kernels are represented by the following matrices:
      _               _                   _                _
     |                 |                 |                  |
     | 1.0   0.0  -1.0 |                 |  1.0   2.0   1.0 |
Gx = | 2.0   0.0  -2.0 |    and     Gy = |  0.0   0.0   0.0 |
     | 1.0   0.0  -1.0 |                 | -1.0  -2.0  -1.0 |
     |_               _|                 |_                _|
"""

# Here we define the matrices associated with the Sobel filter
Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
# we need to know the shape of the input grayscale image
[rows, columns] = np.shape(input_image)
# initialization of the output image array (all elements are 0)
sobel_filtered_image = np.zeros(shape=(rows, columns))

# Now we "sweep" the image in both x and y directions and compute the output
cpu_start_time = time()  # Get the CPU start time
for i in range(rows - 2):
    for j in range(columns - 2):
        # x direction
        gx = np.sum(np.multiply(Gx, input_image[i:i + 3, j:j + 3]))
        # y direction
        gy = np.sum(np.multiply(Gy, input_image[i:i + 3, j:j + 3]))
        # calculate the "hypotenuse"
        sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)/2

cpu_end_time = time()  # Get the CPU end time
# Print how long the CPU took
print("CPU Time: {0} s".format(cpu_end_time - cpu_start_time))
# Save the filtered image in destination path
imsave("results/sobelFiltered.png", sobel_filtered_image.astype(np.uint8))
