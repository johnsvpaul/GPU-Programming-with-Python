
import numpy as np
import pyopencl as cl
from PIL import Image
from time import time

# retreiving kernel


def getKernel(krnl):
    kernel = open(krnl).read()
    return kernel

# initialising kernel and setting up buffers


def findedges(p, d, image):

    data = np.asarray(image).astype(np.int32)

    platform = cl.get_platforms()[p]  # Get platforms, both GPU and CPU
    device = platform.get_devices()[d]
    cntx = cl.Context([device])  # get context
    queue = cl.CommandQueue(cntx)  # queue instantiation

    mf = cl.mem_flags
    # allocating memory for input buffer
    im = cl.Buffer(cntx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    # allocating memory for output buffer
    out = cl.Buffer(cntx, mf.WRITE_ONLY, data.nbytes)

    prgm = cl.Program(cntx, getKernel('OpenCl/edge.cl') %
                      (data.shape[1], data.shape[0])).build()

    gpu_start_time = time()  # getting GPU start time

    # name of function in edge.c file
    prgm.detectedge(queue, data.shape, None, im, out)

    result = np.empty_like(data)  # Create an empty array the same size as data

    # Read back data from GPU memory into result
    cl.enqueue_copy(queue, result, out)
    gpu_end_time = time()

    result = result.astype(np.uint8)
    print(result)

    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))

    img = Image.fromarray(result)

    img.save('yodaedge.png')  # save image


if __name__ == '__main__':

    image = Image.open('dataset/yoda.png')
    # (1,0) is my platform 1, device 0 = "AMD gpu"
    # (0,0) for intel processor
    findedges(1, 0, image)
