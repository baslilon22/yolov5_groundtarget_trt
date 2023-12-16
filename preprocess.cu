#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#ifndef __CUDACC__
#define __CUDACC__
#include "cuda_texture_types.h"
#include "texture_indirect_functions.h"
#include "texture_fetch_functions.h"
#endif


texture<unsigned char, 2> texIn;
cudaArray *texarray;
cudaChannelFormatDesc desc;
#include"preprocess.h"

__global__ void ToChannelLast(float* data_output,unsigned char* input_image,int imageStep,int INPUT_H, int INPUT_W,int batch_index)

{
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < INPUT_H && col < INPUT_W) 
    {
        unsigned char* uc_pixel = input_image + row * imageStep;
        int i = row*INPUT_W+col;
        uc_pixel += 3*col;

        int baseIndex = batch_index * 3 * INPUT_H * INPUT_W + i;
        data_output[baseIndex] = (float)uc_pixel[2] / 255.0;
        data_output[baseIndex + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
        data_output[baseIndex + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
        // uc_pixel += 3;
        // ++i;
    }
}

void ToChannelLast_GPU(float* data_output,unsigned char* input_image,int imageStep,int INPUT_H, int INPUT_W,int batch_index,int batch_size)
{
    unsigned char* d_input_image;
    cudaMalloc((void**)&d_input_image, 3*INPUT_W*INPUT_H * sizeof(unsigned char));
    cudaMemcpy(d_input_image, input_image, 3*INPUT_W*INPUT_H*sizeof(unsigned char), cudaMemcpyHostToDevice);

    float* d_data_out;
    cudaMalloc((void**)&d_data_out, batch_size * 3 * INPUT_H * INPUT_W * sizeof(float));

    dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((INPUT_W + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (INPUT_H + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
    ToChannelLast<<<blocksPerGrid,threadsPerBlock>>>(d_data_out,d_input_image,imageStep,INPUT_H, INPUT_W,batch_index);

    cudaMemcpy(data_output, d_data_out, batch_size * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data_out);
    cudaFree(d_input_image);
    cudaDeviceSynchronize();

}



