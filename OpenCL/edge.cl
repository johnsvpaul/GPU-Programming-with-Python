__kernel void detectedge(__global int *im,__global int *out){
      int j = get_global_id(1);
      int i = get_global_id(0);

      
      int width = %d;
      int rown = %d;
      int value;

        
              value = -im[(i)*width + j] -  0* im[(i)*width + j+1] + im[(i)*width + j+2]
                      -2*im[(i+1)*width + j] +  0*im[(i+1)*width + j+1] + 2*im[(i+1)*width + j+2]
                      -im[(i+2)*width + j] -  0*im[(i+2)*width + j+1] + im[(i+2)*width + j+2];

              value = (value < 0   ? 0   : value);
              value = (value > 255 ? 255 : value);
              out[i*width + j] = value;

  }
