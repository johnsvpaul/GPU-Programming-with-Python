from platform import platform
import pyopencl as cl
import numpy as np

from imageio import imread, imsave
from time import time  # Import time tools

image = "1L"  # image name
noiseLevel = ["30NOISE", "50"]  # percent of noise

# Read in image
img = imread("dataset/"+noiseLevel[0]+"/"+image+".png").astype(np.float32)

# retreiving kernel


def getKernel(krnl):
    kernel = open(krnl).read()
    return kernel


# Get platforms, both CPU and GPU
plat = cl.get_platforms()

CPU = plat[0].get_devices()
# checks to see if GPU exists
try:
    GPU = plat[1].get_devices()

except IndexError:
    GPU = "none"

# Create context for GPU/CPU
if GPU != "none":  # if GPU does exist then create context for GPU
    ctx = cl.Context(GPU)
else:  # if GPU does not exist then create context for CPU
    ctx = cl.Context(CPU)

# Create queue for each kernel execution


# ctx = cl.create_some_context()  # use this if you want it to prompt you to select platform/device

# Instantiate a Queue with profiling (timing) enabled
queue = cl.CommandQueue(
    ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags


# Allocate memory for variables on the device
img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
result_g = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
width_g = cl.Buffer(
    ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1])
)
height_g = cl.Buffer(
    ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0])
)
# Kernel function instantiation
program = cl.Program(ctx, getKernel('OpenCl/median.cl')
                     ).build()  # Compile the device program
gpu_start_time = time()  # Get the GPU start time

event = program.medianFilter(queue, img.shape, None, img_g,
                             result_g, width_g, height_g)  # Enqueue the GPU sum program
event.wait()  # Wait until the event finishes XXX

# Calculate the time it took to execute the kernel
elapsed = 1e-9*(event.profile.end - event.profile.start)
# Print the time it took to execute the kernel
print("GPU Kernel Time: {0} s".format(elapsed))

# Create an empty array the same size as array img
removed_noise = np.empty_like(img)

# Read back the data from GPU memory into array img

cl.enqueue_copy(queue, removed_noise, result_g).wait()

gpu_end_time = time()  # Get the GPU end time

# Print the time the GPU program took, including both memory copies (read+write to gpu)
print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))


# for platform in cl.get_platforms():
#     for device in platform.get_devices():
#         print("===============================================================")
#         print("Platform name:", platform.name)
#         print("Platform profile:", platform.profile)
#         print("Platform vendor:", platform.vendor)
#         print("Platform version:", platform.version)
#         print("---------------------------------------------------------------")
#         print("Device name:", device.name)
#         print("Device type:", cl.device_type.to_string(device.type))
#         print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
#         print("Device max clock speed:", device.max_clock_frequency, 'MHz')
#         print("Device compute units:", device.max_compute_units)

# save image
imsave("results/OpenCL" +
       noiseLevel[0]+image+".png", removed_noise.astype(np.uint8))
