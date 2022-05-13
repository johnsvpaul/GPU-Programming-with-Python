import numpy as np
import pyopencl as cl
from imageio import imread, imsave
from time import time

# retreiving kernel


def getKernel(krnl):
    kernel = open(krnl).read()
    return kernel


def findedges(p, d, img):  # p = platform, d= device, img=image as an array

    platform = cl.get_platforms()[p]  # Get platforms, both GPU and CPU
    device = platform.get_devices()[d]
    ctx = cl.Context([device])  # get context
    # Instantiate a Queue with profiling (timing) enabled
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    mf = cl.mem_flags
    # allocating memory for input buffer
    im = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    # allocating memory for output buffer
    out = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
    # Compile the device program
    prgm = cl.Program(ctx, getKernel('OpenCl/edge.cl') %
                      (img.shape[1], img.shape[0])).build()

    gpu_start_time = time()  # getting GPU start time
    # enqueue the kernel function in edge.cl to the GPU
    event = prgm.detectedge(queue, img.shape, None, im, out)
    event.wait()
    # Calculate the time it took to execute the kernel
    elapsed = 1e-9*(event.profile.end - event.profile.start)

    # Print the time it took to execute the kernel
    print("GPU Kernel Time: {0} s".format(elapsed))

    result = np.empty_like(img)  # Create an empty array the same size as data

    # Read back data from GPU memory into result
    cl.enqueue_copy(queue, result, out)

    gpu_end_time = time()
    # Print the time the GPU program took, including both memory copies (read+write to gpu)
    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))
    # save image
    imsave("results/"+image+".png", result.astype(np.uint8))


if __name__ == '__main__':

    image = "coins"  # image name
    # Read in image
    img = imread("dataset/"+image+".png").astype(np.float32)
    # (1,0) is my platform 1, device 0 = "AMD gpu"
    # (0,0) for intel processor
    findedges(1, 0, img)
