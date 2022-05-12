
import numpy as np
import pyopencl as cl
from PIL import Image
from time import time


def getKernel(krnl):
    kernel = open(krnl).read()
    return kernel


def findedges(p, d, image):

    data = np.asarray(image).astype(np.int32)

    platform = cl.get_platforms()[p]
    device = platform.get_devices()[d]
    cntx = cl.Context([device])
    queue = cl.CommandQueue(cntx)

    mf = cl.mem_flags
    im = cl.Buffer(cntx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    out = cl.Buffer(cntx, mf.WRITE_ONLY, data.nbytes)

    prgm = cl.Program(cntx, getKernel('OpenCl/edge.c') %
                      (data.shape[1], data.shape[0])).build()

    prgm.detectedge(queue, data.shape, None, im, out)

    result = np.empty_like(data)

    cl.enqueue_copy(queue, result, out)
    result = result.astype(np.uint8)
    print(result)

    img = Image.fromarray(result)
    # img.show()
    img.save('yodaedge.png')


if __name__ == '__main__':

    image = Image.open('dataset/yoda.png')
    # (1,0) is my platform 1, device 0 = "AMD gpu"
    # (0,0) for intel processor
    findedges(1, 0, image)
