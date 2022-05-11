from platform import platform
import pyopencl as cl
import numpy as np

from imageio import imread, imsave
from time import time  # Import time tools

image = "girl"
noiseLevel = ["30", "50"]

# Read in image
img = imread("dataset/"+noiseLevel[0]+"/"+image+".png").astype(np.float32)


# Kernel function
src = """
void sort(int *a, int *b, int *c) {
   int swap;
   if(*a > *b) {
      swap = *a;
      *a = *b;
      *b = swap;
   }
   if(*a > *c) {
      swap = *a;
      *a = *c;
      *c = swap;
   }
   if(*b > *c) {
      swap = *b;
      *b = *c;
      *c = swap;
   }
}
__kernel void medianFilter(
    __global float *img, __global float *result, __global int *width, __global
    int *height)
{
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    // Keeping the edge pixels the same
    if( posx == 0 || posy == 0 || posx == w-1 || posy == h-1 )
    {
        result[i] = img[i];
    }
    else
    {
        int pixel00, pixel01, pixel02, pixel10, pixel11, pixel12, pixel20,
            pixel21, pixel22;
        pixel00 = img[i - 1 - w];
        pixel01 = img[i- w];
        pixel02 = img[i + 1 - w];
        pixel10 = img[i - 1];
        pixel11 = img[i];
        pixel12 = img[i + 1];
        pixel20 = img[i - 1 + w];
        pixel21 = img[i + w];
        pixel22 = img[i + 1 + w];
        //sort the rows
        sort( &(pixel00), &(pixel01), &(pixel02) );
        sort( &(pixel10), &(pixel11), &(pixel12) );
        sort( &(pixel20), &(pixel21), &(pixel22) );
        //sort the columns
        sort( &(pixel00), &(pixel10), &(pixel20) );
        sort( &(pixel01), &(pixel11), &(pixel21) );
        sort( &(pixel02), &(pixel12), &(pixel22) );
        //sort the diagonal
        sort( &(pixel00), &(pixel11), &(pixel22) );
        // median is the the middle value of the diagonal
        result[i] = pixel11;
    }
}
"""
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
program = cl.Program(ctx, src).build()  # Compile the device program
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

# Print the time the GPU program took, including both memory copies
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
imsave("results/OpenCLfiltered" +
       noiseLevel[0]+image+".png", removed_noise.astype(np.uint8))
