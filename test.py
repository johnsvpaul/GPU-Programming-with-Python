# Test the speed of your PyOpenCL program
from time import time  # Import time tools

import pyopencl as cl  # Import the OpenCL GPU computing API
import numpy as np  # Import number tools

a = np.random.rand(1000).astype(np.float32)  # Create a random array to add
b = np.random.rand(1000).astype(np.float32)  # Create a random array to add


def cpu_array_sum(a, b):  # Sum two arrays on the CPU
    c_cpu = np.empty_like(a)  # Create the destination array
    cpu_start_time = time()  # Get the CPU start time
    for i in range(1000):
        for j in range(1000):  # 1000 times add each number and store it
            # This add operation happens 1,000,000 times XXX
            c_cpu[i] = a[i] + b[i]
    cpu_end_time = time()  # Get the CPU end time
    # Print how long the CPU took
    print("CPU Time: {0} s".format(cpu_end_time - cpu_start_time))
    return c_cpu  # Return the sum of the arrays


def gpu_array_sum(a, b):
    context = cl.create_some_context()  # Initialize the Context
    # Instantiate a Queue with profiling (timing) enabled
    queue = cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY |
                         cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY |
                         cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
    # Create three buffers (plans for areas of memory on the device)
    c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)
    program = cl.Program(context, """
    __kernel void sum(__global const float *a, __global const float *b, __global float *c)
    {
        int i = get_global_id(0);
        int j;
        for(j = 0; j < 1000; j++)
        {
            c[i] = a[i] + b[i];
        }
    }""").build()  # Compile the device program
    gpu_start_time = time()  # Get the GPU start time
    # Enqueue the GPU sum program XXX
    event = program.sum(queue, a.shape, None, a_buffer, b_buffer, c_buffer)
    event.wait()  # Wait until the event finishes XXX
    # Calculate the time it took to execute the kernel
    elapsed = 1e-9*(event.profile.end - event.profile.start)
    # Print the time it took to execute the kernel
    print("GPU Kernel Time: {0} s".format(elapsed))
    c_gpu = np.empty_like(a)  # Create an empty array the same size as array a
    # Read back the data from GPU memory into array c_gpu
    cl.enqueue_copy(queue, c_buffer, c_gpu).wait()
    gpu_end_time = time()  # Get the GPU end time
    # Print the time the GPU program took, including both memory copies
    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))
    return c_gpu  # Return the sum of the two arrays


cpu_array_sum(a, b)  # Call the function that sums two arrays on the CPU
gpu_array_sum(a, b)  # Call the function that sums two arrays on the GPU