
__kernel void detectedge(__global int *im,__global int *out){

      int j = get_global_id(0);
      int i = get_global_id(1);
      

      int width = %d;
      int rown = %d;

      int value;
      int Gx = 0;
      int Gy = 0;
      
if (i >= 1 && i < (width - 1) && j >= 1 && j < rown - 1)
	{
      int i00 = (im[(i - 1) + (j - 1) * width]);
      int i10 = (im[i + (j - 1) * width]);
      int i20 = (im[(i + 1) + (j - 1) * width]);
      int i01 = (im[(i - 1) + j * width]);
      int i11 = (im[i + j * width]);
      int i21 = (im[(i + 1) + j * width]);
      int i02 = (im[(i - 1) + (j + 1) * width]);
      int i12 = (im[i + (j + 1) * width]);
      int i22 = (im[(i + 1) + (j + 1) * width]);

//The kernels Gx and Gy can be thought of as a differential operation in the "input_image" array in the directions x and y 
//respectively. These kernels are represented by the following matrices:
//      _               _                   _                _
//     |                 |                 |                  |
//     | 1.0   0.0  -1.0 |                 |  1.0   2.0   1.0 |
//Gx = | 2.0   0.0  -2.0 |    and     Gy = |  0.0   0.0   0.0 |
//     | 1.0   0.0  -1.0 |                 | -1.0  -2.0  -1.0 |
//     |_               _|                 |_                _|
      
      Gx = i00 - i20 + 2 * i01 - 2 * i21 + i02 - i22;
      Gy = i00 + 2 * i10 + i20 - i02 - 2 * i12 - i22;
     

      out[i+j*width] = hypot((float)Gy,Gx)/2;
      
      
        
    }
  }
