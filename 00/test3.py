import cv2
import numpy as np
from time import time  # Import time tools
img = cv2.imread('yoda.png')
cpu_start_time = time()  # Get the CPU start time
median = cv2.medianBlur(img, 7)
cpu_end_time = time()  # Get the CPU end time
# Print how long the CPU took
print("CPU Time: {0} s".format(cpu_end_time - cpu_start_time))
compare = np.concatenate((img, median), axis=1)  # side by side comparison

cv2.imshow('img', compare)
cv2.waitKey(0)
cv2.destroyAllWindows
