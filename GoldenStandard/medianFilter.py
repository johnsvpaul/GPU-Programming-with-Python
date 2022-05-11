import numpy as np
from time import time  # Import time tools
from imageio import imread, imsave


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data), len(data[0])))
    cpu_start_time = time()  # Get the CPU start time
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    cpu_end_time = time()  # Get the CPU end time
    # Print how long the CPU took
    print("CPU Time: {0} s".format(cpu_end_time - cpu_start_time))
    return data_final


def main():
    image = "girl"
    noiseLevel = ["30", "50"]
    img = imread("dataset/"+noiseLevel[0]+"/"+image+".png").astype(np.float32)
    removed_noise = median_filter(img, 3).astype(np.uint8)
    imsave("results/filtered"+noiseLevel[0]+image+".png", removed_noise)


main()
