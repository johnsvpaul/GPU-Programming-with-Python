import numpy as np
import time
from imageio import imread, imsave


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data), len(data[0])))
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
    return data_final


start = time.perf_counter()


def main():

    img = imread("bridge.png").astype(np.float32)
    removed_noise = median_filter(img, 3)
    imsave("yolo.png", removed_noise)


main()
end = time.perf_counter()
print(f'Finished in {round(end-start, 2)} second(s)')
