#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TILE_WIDTH 256

/* unfinished */
__global__ void MatrixMultipleTile(int *InputImage,int width,int height,int *filter,int filterWidth,int *featureMap)
{
    __shared__ int tileImage[TILE_WIDTH][TILE_WIDTH];
    
    int threadID=blockIdx.x*blockDim.x+threadIdx.x;

    int bx=blockIdx.x;
    int by=blockIdx.y;
    int tx=blockIdx.x;
    int ty=blockIdx.y;

    int Row=by*TILE_WIDTH+ty;
    int Col=bx*TILE_WIDTH+tx;
    
    tileImage[tx][ty]=InputImage[Row*width+Col];
    __syncthreads();
    int value=0;
    for(int i=0;i<9;++i)
    {
        value+=5;
    }

}
/*unfinished*/
__global__ void MatrixMultiple(int *InputImage,int width,int height,int *filter,int filterWidth,int *featureMap)
{
    /* get global row col */
    int Row=blockIdx.y*TILE_WIDTH+threadIdx.y;
    int Col=blockIdx.x*TILE_WIDTH+threadIdx.x;
    int value=0;
    for(int i=0;i<filterWidth;i++)
    {
        for(int j=0;j<filterWidth;j++)
        {
            value=filter[i*filterWidth+j]* InputImage[(Row+i)*width+Col+j];
        }
    }
    featureMap[threadID]=value;
}

void convolution(int *InputImage,int width,int height,int *filter,int filterWidth,int *featureMap)
{
    
    int *featureMapd,*InputImaged,*filterd;
    int blockNum,x,y,featureMapWidthx,featureMapWidthy;
    cudaMalloc(&InputImaged,width*height*sizeof(int));
    cudaMemcpy(InputImaged,InputImage,width*height*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&filter,filterWidth*filterWidth*sizeof(int));
    cudaMemcpy(filterd,filter,filterWidth*filterWidth*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&featureMapd,(width-filterWidth+1)*(height-filterWidth+1)*sizeof(int));

    featureMapWidthx=height-filterWidth+1;
    featureMapWidthy=width-filterWidth+1;

    x=(featureMapWidthx+TILE_WIDTH-1)/TILE_WIDTH;
    y=(featureMapWidthy+TILE_WIDTH-1)/TILE_WIDTH;
    
    dim3 dimGrid(x,y);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
    MatrixMultiple<<<dimGrid,dimBlock>>>(InputImaged,width,height,filterd,filterWidth,featureMapd);
    
    cudaMemcpy(featureMap,featureMapd,size,cudaMemcpyDeviceToHost);
    
    cudaFree(featureMapd);cudaFree(InputImaged);cudaFree(filterd);
}
