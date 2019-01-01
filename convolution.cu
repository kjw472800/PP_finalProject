#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "image.h"
#include "filter.h"

#define TILE_WIDTH 16
#define FSize 9
//void convolution(int *InputImage,int width,int height,int *filter,int filterWidth,,int padding,int *result);
using namespace std;

__global__ void MatrixMultiple(int *InputImage,int width,int height,int *filter,int filterWidth,int *featureMap);
int* pad_array(int* input, int width, int height, int padding);
__constant__ int cntfilterd[FSize];


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* one feature map element map to one thread*/
__global__ void MatrixMultiple(int *InputImage,int width,int height,int *filter,int filterWidth,int *featureMap)
{
    /* get global row col */
    int Row=blockIdx.y*TILE_WIDTH+threadIdx.y;
    int Col=blockIdx.x*TILE_WIDTH+threadIdx.x;
    int value=0;
    int feathreMapwidth=width-filterWidth+1;
    if(Row*width+Col<width*height)
    {
        for(int i=0;i<filterWidth;i++)
        {
            for(int j=0;j<filterWidth;j++)
            {
                value+=filter[i*filterWidth+j]* InputImage[(Row+i)*width+Col+j];
            }
        }
        //printf("%d %d\n",Row*width+Col,value);

        featureMap[feathreMapwidth*Row+Col]=value;
    }
    //printf("%d %d\n",Row*width+Col,value);
}
__global__ void cntMatrixMultiple(int *InputImage,int width,int height,int filterWidth,int *featureMap)
{
    /* get global row col */
    int Row=blockIdx.y*TILE_WIDTH+threadIdx.y;
    int Col=blockIdx.x*TILE_WIDTH+threadIdx.x;
    int value=0;
    int feathreMapwidth=width-filterWidth+1;
    if(Row*width+Col<width*height)
    {
        for(int i=0;i<filterWidth;i++)
        {
            for(int j=0;j<filterWidth;j++)
            {
                value+=cntfilterd[i*filterWidth+j]* InputImage[(Row+i)*width+Col+j];
            }
        }
        //printf("%d %d\n",Row*width+Col,value);

        featureMap[feathreMapwidth*Row+Col]=value;
    }
    //printf("%d %d\n",Row*width+Col,value);
}

int * cntconvolution(int *InputImage,int width,int height,int *filter,int filterWidth,int padding,int *result)
{

    int *featureMapd,*InputImaged,*filterd,*featureMap;
    int x,y,featureMapWidth,featureMapHeight;
    int originImageSize=width*height*sizeof(int);
    int filterSize=filterWidth*filterWidth*sizeof(int);
    int feathreMapSize;
    cout<<"in constant convolution"<<endl;
    featureMapHeight=height-filterWidth+1; //feature map's width = origin width-featureWidth+1
    featureMapWidth=width-filterWidth+1;
    feathreMapSize=featureMapHeight*featureMapWidth*sizeof(int);
    featureMap= new int[feathreMapSize];

    /*for(int i=0;i<width*height;i++)
    {
        cout<<i<<" "<<InputImage[i]<<endl;
    }*/
    cudaMalloc(&InputImaged,originImageSize);
    cudaMemcpy(InputImaged,InputImage,originImageSize,cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(cntfilterd, filter, sizeof(int) * FSize);

    cudaMalloc(&featureMapd,feathreMapSize);

    cout<<"in"<<endl;
    // determine which blocks
    x=(featureMapWidth+TILE_WIDTH-1)/TILE_WIDTH;
    y=(featureMapHeight+TILE_WIDTH-1)/TILE_WIDTH;

    cout<<x<<" "<<y<<endl;
    dim3 dimGrid(x,y);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

    cntMatrixMultiple<<<dimGrid,dimBlock>>>(InputImaged,width,height,filterWidth,featureMapd);

    cudaMemcpy(featureMap,featureMapd,feathreMapSize,cudaMemcpyDeviceToHost);
    /*for(int i=0;i<featureMapHeight*featureMapWidth;i++)
    {
        cout<<i<<" "<<featureMap[i]<<endl;
    }*/
    cudaFree(featureMapd);cudaFree(InputImaged);cudaFree(filterd);

    return result=pad_array(featureMap,featureMapWidth,featureMapHeight,padding);

    /*for(int i=0;i<width*height;i++)
    {
        cout<<i<<" "<<result[i]<<endl;
    }*/
}

int * convolution(int *InputImage,int width,int height,int *filter,int filterWidth,int padding,int *result)
{

    int *featureMapd,*InputImaged,*filterd,*featureMap;
    int x,y,featureMapWidth,featureMapHeight;
    int originImageSize=width*height*sizeof(int);
    int filterSize=filterWidth*filterWidth*sizeof(int);
    int feathreMapSize;
    cout<<"in normal convolution"<<endl;
    featureMapHeight=height-filterWidth+1; //feature map's width = origin width-featureWidth+1
    featureMapWidth=width-filterWidth+1;
    feathreMapSize=featureMapHeight*featureMapWidth*sizeof(int);
    featureMap= new int[feathreMapSize];

    /*for(int i=0;i<width*height;i++)
    {
        cout<<i<<" "<<InputImage[i]<<endl;
    }*/
    cudaMalloc(&InputImaged,originImageSize);
    cudaMemcpy(InputImaged,InputImage,originImageSize,cudaMemcpyHostToDevice);

    cudaMalloc(&filterd,filterSize);
    cudaMemcpy(filterd,filter,filterSize,cudaMemcpyHostToDevice);

    cudaMalloc(&featureMapd,feathreMapSize);

    cout<<"in"<<endl;
    // determine which blocks
    x=(featureMapWidth+TILE_WIDTH-1)/TILE_WIDTH;
    y=(featureMapHeight+TILE_WIDTH-1)/TILE_WIDTH;

    cout<<x<<" "<<y<<endl;
    dim3 dimGrid(x,y);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

    MatrixMultiple<<<dimGrid,dimBlock>>>(InputImaged,width,height,filterd,filterWidth,featureMapd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(featureMap,featureMapd,feathreMapSize,cudaMemcpyDeviceToHost);
    /*for(int i=0;i<featureMapHeight*featureMapWidth;i++)
    {
        cout<<i<<" "<<featureMap[i]<<endl;
    }*/
    cudaFree(featureMapd);cudaFree(InputImaged);cudaFree(filterd);

    return result=pad_array(featureMap,featureMapWidth,featureMapHeight,padding);

    /*for(int i=0;i<width*height;i++)
    {
        cout<<i<<" "<<result[i]<<endl;
    }*/
}

int main(int argc, char *argv[])
{
	if(argc < 3) {
        printf("Usage: ./serial_m <image_filename> <filter_filename>\n");
        return 0;
    }

    int *image_r, *image_g, *image_b;
    int image_width, image_height;

    if(read_image(argv[1], &image_r, &image_g, &image_b, &image_width, &image_height) < 0) {
        printf("Error: can not open %s\n", argv[1]);
        return -1;
    }

    //----------------------------------------------------------------------------------------
    int num_filters;
    int *fil_size;
    int **fil_matrix;
    load_filter(argv[2], &num_filters, &fil_matrix, &fil_size);

    printf("\n******************************************\n");
    printf("Do convolution\n");

    int *conv_r, *conv_g, *conv_b;
    for(int i = 0; i < num_filters; i++)
    {
        printf("filter %d:\n", i);
        print_filter(fil_matrix[i], fil_size[i]);

        conv_r=cntconvolution(image_r,image_width,image_height,fil_matrix[i],fil_size[i],1,conv_r);
        conv_g=cntconvolution(image_g,image_width,image_height,fil_matrix[i],fil_size[i],1,conv_g);
        conv_b=cntconvolution(image_b,image_width,image_height,fil_matrix[i],fil_size[i],1,conv_b);
        /*cout<<"print"<<endl;
        cout<<i<<endl;
        cout<<conv_r[0]<<endl;
        for(int i=0;i<image_width*image_height;i++)
        {
            cout<<i<<" "<<conv_r[i]<<endl;
        }*/
        show_image(conv_r, conv_g, conv_b, image_width, image_height);

        free_image(conv_r, conv_g, conv_b);
    }

    printf("Convolution done.\n");
    printf("******************************************\n");

    free_image(image_r, image_g, image_b);
    free_filter(num_filters, fil_matrix, fil_size);
    printf("\ndone.\n");
    return 0;
}


int* pad_array(int* input, int width, int height, int padding) {
    int new_width = width+2*padding;
    int new_height = height+2*padding;
    int* padded_array = new int [new_width * new_height * sizeof(int)];
    memset (padded_array, 0, new_width * new_height * sizeof(int));

    for(int i = padding; i < new_height-padding; ++i) {
        for(int j = padding; j < new_width-padding; ++j) {
            *(padded_array+i*new_width+j) = *(input+(i-padding)*width+(j-padding));
        }
    }

    return padded_array;
}

/* unfinished */

/*
__global__ void shareMatrixMultiple(int *InputImage,int width,int height,int *filter,int filterWidth,int *featureMap)
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
*/
/*unfinished*/
