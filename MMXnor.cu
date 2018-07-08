#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <cuda_runtime.h>
using namespace std;

void MatrixRandBin(float *mat,int rows,int cols){
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            if((float)rand()/RAND_MAX>0.5){
                mat[i*cols+j]=1.0f;
            }else{
                mat[i*cols+j]=-1.0f;
            }

        }
    }
}

void MatrixPrint(float *mat,int rows,int cols){
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            cout<<setw(2)<<mat[i*cols+j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

void MatrixPrintD(int *mat,int rows,int cols){
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            cout<<setw(2)<<mat[i*cols+j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}


float MatrixCompare(float *a,float *b,int rows,int cols){
    float err=0;
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            err+=abs(a[i*cols+j]-b[i*cols+j]);  
        }
    }
    return err;
}

void MatrixMul_host(float *a,int a_rows,int a_cols,float *b,int b_rows,int b_cols,float *c)
{
    for(int i = 0; i < a_rows; i++) {
        for(int j = 0; j < b_cols; j++) {
            float t = 0;
            for(int k = 0; k < b_rows; k++) {
                t += a[i * a_cols + k] * b[k * b_cols + j];
            }
            c[i * b_cols + j] = t;
        }
    }
}

//horizontal
__global__ void AMatrix2Bin(float *a,int *a_bin,int a_rows,int pitch_a,int pitch_a_bin,int MaxBS,int BINSIZE){
    int tix=threadIdx.x;
    // int tiy=threadIdx.y;
    int bix=blockIdx.x;
    // int biy=blockIdx.y;
    int bdx=blockDim.x;
    // int bdy=blockDim.y;
    int gdx=gridDim.x;
    // int gdy=gridDim.y;


    int maxThreads=MaxBS*a_rows;
    for(int id = bix*bdx+tix; id < maxThreads; id+=gdx*bdx) {
        int rid=id/MaxBS;
        int cid=id%MaxBS;

        int Integer=0;
        int base=1;
        for (int i=0;i<BINSIZE;i++){
            if (a[rid*pitch_a+(cid+1)*BINSIZE-1-i]==1.f){
                Integer+=base;
            }
            base=base<<1;
        }

        a_bin[rid*pitch_a_bin+cid]=Integer;
    }

}
//vetical
__global__ void BMatrix2Bin(float *b,int *b_bin,int b_cols,int pitch_b,int pitch_b_bin,int MaxBS,int BINSIZE){
    int tix=threadIdx.x;
    // int tiy=threadIdx.y;
    int bix=blockIdx.x;
    // int biy=blockIdx.y;
    int bdx=blockDim.x;
    // int bdy=blockDim.y;
    int gdx=gridDim.x;
    // int gdy=gridDim.y;

    int maxThreads=MaxBS*b_cols;
    for(int id = bix*bdx+tix; id < maxThreads; id+=gdx*bdx) {
        int cid=id/MaxBS;
        int rid=id%MaxBS;

        int Integer=0;
        int base=1;
        for (int i=0;i<BINSIZE;i++){
            if (b[((rid+1)*BINSIZE-1-i)*pitch_b+cid]==1.f){
                Integer+=base;
            }
            base=base<<1;
        }

        b_bin[rid*pitch_b_bin+cid]=Integer;
    }

}

__device__ unsigned char __popcount_tab_device[256];//__constant__ is slower than __device__
__device__ int popcount (int x) {
  return __popcount_tab_device[(x >>  0) & 0xff]  
  + __popcount_tab_device[(x >>  8) & 0xff]  
  + __popcount_tab_device[(x >> 16) & 0xff] 
  + __popcount_tab_device[(x >> 24) & 0xff];
}

__global__ void MatrixMulXnor(int *a,int *b,int a_rows,int a_cols,
    int b_cols,float *result,int pitch_a,int pitch_b,
    int pitch_result,int BINSIZE,int RealMidSize){

    int tix=threadIdx.x;
    // int tiy=threadIdx.y;
    int bix=blockIdx.x;
    // int biy=blockIdx.y;
    int bdx=blockDim.x;
    // int bdy=blockDim.y;
    int gdx=gridDim.x;
    // int gdy=gridDim.y;

    int rest=(BINSIZE*a_cols-RealMidSize);

    for(int j=tix;j<b_cols;j+=bdx){
        // printf("i=%d ; j=%d\n",i,j);
        int sum=0;
        for(int k=0;k<a_cols;k++){
            int bin=(a[bix*pitch_a+k]^b[k*pitch_b+j]);
            int negnum=popcount(bin);
            int posnum=BINSIZE-negnum;
            //calculate ignores the rest of BINSIZE if the Matsize cant devided by BINSIZE ,it can cause err
            //(10/00)'(01/00) should be 0000 but it is 0011,so 1+1 is trash in the result.and it can cause a_rows*b_cols times. 
            sum+=(posnum-negnum);
        }
        result[bix*pitch_result+j]=sum-rest;
    }
    


}


void MatrixMul_device(float *a,float *b,int a_rows,int a_cols,int b_cols,float *result){

    int BINSIZE=30;//size of bin2int, 32 means 0000 0000 0000 0000 0000 0000 0000 0000
    int MaxBS=(a_cols-1)/BINSIZE+1;
    int a_cols_Copysize=MaxBS*BINSIZE;
    dim3 BS_BIN(512,1,1);
    dim3 GS_BIN(6,1,1);
    

    float *a_device;//a_rows * a_cols_Copysize
    float *b_device;//a_cols_Copysize * b_cols
    size_t pitch_a_device, pitch_b_device;
    cudaMallocPitch((void**)&a_device , &pitch_a_device , sizeof(float) *a_cols_Copysize , a_rows);
    cudaMallocPitch((void**)&b_device , &pitch_b_device , sizeof(float) *b_cols , a_cols_Copysize);
    cudaMemset(a_device, 0, pitch_a_device * a_rows);
    cudaMemset(b_device, 0, pitch_b_device * a_cols_Copysize);
    cudaMemcpy2D(a_device,pitch_a_device,a,sizeof(float) *a_cols ,sizeof(float) *a_cols, a_rows,cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(b_device,pitch_b_device,b,sizeof(float) *b_cols ,sizeof(float) *b_cols, a_cols,cudaMemcpyDeviceToDevice);

//check oringin
    // float *a_host;
    // float *b_host;
    // a_host = (float*) malloc(sizeof(float) * a_cols_Copysize * a_rows);
    // b_host = (float*) malloc(sizeof(float) * b_cols * a_cols_Copysize);
    // cudaMemcpy2D(a_host,sizeof(float) *a_cols_Copysize, a_device,pitch_a_device,sizeof(float) *a_cols_Copysize , a_rows,cudaMemcpyDeviceToHost);
    // cudaMemcpy2D(b_host,sizeof(float) *b_cols, b_device,pitch_b_device,sizeof(float) *b_cols , a_cols_Copysize,cudaMemcpyDeviceToHost);
    // MatrixPrint(a_host,a_rows,a_cols_Copysize);
    // MatrixPrint(b_host,a_cols_Copysize,b_cols);

    int *a_device_bin;
    int *b_device_bin;
    size_t pitch_a_device_bin, pitch_b_device_bin;
    cudaMallocPitch((void**)&a_device_bin , &pitch_a_device_bin , sizeof(int) *MaxBS , a_rows);
    cudaMallocPitch((void**)&b_device_bin , &pitch_b_device_bin , sizeof(int) *b_cols , MaxBS);

    AMatrix2Bin<<<GS_BIN,BS_BIN>>>(a_device , a_device_bin , a_rows , 
        pitch_a_device/sizeof(float) , pitch_a_device_bin/sizeof(int) , MaxBS , BINSIZE);
    BMatrix2Bin<<<GS_BIN,BS_BIN>>>(b_device , b_device_bin , b_cols , 
        pitch_b_device/sizeof(float) , pitch_b_device_bin/sizeof(int) , MaxBS , BINSIZE);

//check bin
    // int *a_host_bin;
    // int *b_host_bin;
    // a_host_bin = (int*) malloc(sizeof(int) *MaxBS * a_rows);
    // b_host_bin = (int*) malloc(sizeof(int) *b_cols * MaxBS);
    // cudaMemcpy2D(a_host_bin,sizeof(int) *MaxBS, a_device_bin,pitch_a_device_bin,sizeof(int) *MaxBS , a_rows ,cudaMemcpyDeviceToHost);
    // cudaMemcpy2D(b_host_bin,sizeof(int) *b_cols, b_device_bin,pitch_b_device_bin,sizeof(int) *b_cols , MaxBS ,cudaMemcpyDeviceToHost);
    // MatrixPrintD(a_host_bin,a_rows,MaxBS);
    // MatrixPrintD(b_host_bin,MaxBS,b_cols);


    float *result_device;//a_rows * b_cols
    size_t pitch_result_device;
    cudaMallocPitch((void**)&result_device , &pitch_result_device , sizeof(float) *b_cols , a_rows);

    const unsigned char __popcount_tab[] = {
      0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,
    };
    cudaMemcpyToSymbol(__popcount_tab_device, __popcount_tab, sizeof(__popcount_tab));

    cudaEvent_t start_device, stop_device;
    float time_device;
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);
    cudaEventRecord( start_device, 0 );

    dim3 BS_MM(32,1,1);
    dim3 GS_MM(1000,1,1);
    MatrixMulXnor<<<GS_MM,BS_MM>>>(a_device_bin , b_device_bin , a_rows , MaxBS , b_cols ,
     result_device , pitch_a_device_bin/sizeof(int) , pitch_b_device_bin/sizeof(int) , 
     pitch_result_device/sizeof(float) , BINSIZE , a_cols);

    cudaEventRecord( stop_device, 0 );
    cudaEventSynchronize( stop_device );
    cudaEventElapsedTime( &time_device, start_device, stop_device );
    cudaEventDestroy( start_device );
    cudaEventDestroy( stop_device );
    cout<<"gputime="<<time_device<<"ms"<<endl;

    cudaMemcpy2D(result,sizeof(float) *b_cols, result_device,pitch_result_device,sizeof(float) *b_cols , a_rows ,cudaMemcpyDeviceToDevice);

    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(a_device_bin);
    cudaFree(b_device_bin);
    cudaFree(result_device);
}

int main(){

//simulate pytorch param
    int Matrixsize=1000;
    float *a_host;
    float *b_host;
    float *result_host;
    a_host = (float*) malloc(sizeof(float) * Matrixsize * Matrixsize);
    b_host = (float*) malloc(sizeof(float) * Matrixsize * Matrixsize);
    result_host = (float*) malloc(sizeof(float) * Matrixsize * Matrixsize);
    srand(0);
    MatrixRandBin(a_host,Matrixsize,Matrixsize);
    MatrixRandBin(b_host,Matrixsize,Matrixsize);
    // cout<<MatrixCopysize<<endl;

    float *a_device;
    float *b_device;
    float *result_device;
    cudaMalloc((void**)&a_device,sizeof(float) *Matrixsize * Matrixsize);
    cudaMalloc((void**)&b_device,sizeof(float) *Matrixsize * Matrixsize);
    cudaMalloc((void**)&result_device,sizeof(float) *Matrixsize * Matrixsize);
    cudaMemcpy(a_device,a_host,sizeof(float) *Matrixsize * Matrixsize,cudaMemcpyHostToDevice);
    cudaMemcpy(b_device,b_host,sizeof(float) *Matrixsize * Matrixsize,cudaMemcpyHostToDevice);


    // MatrixPrint(a_host,Matrixsize,Matrixsize);
    // MatrixPrint(b_host,Matrixsize,Matrixsize);

//run in gpu warp in C code
    MatrixMul_device(a_device,b_device,Matrixsize,Matrixsize,Matrixsize,result_device);

    cudaMemcpy(result_host, result_device,sizeof(float) *Matrixsize * Matrixsize,cudaMemcpyDeviceToHost);
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(result_device);
    // MatrixPrint(result_host,Matrixsize,Matrixsize);

//run in cpu
    float *result_cpu;
    result_cpu = (float*) malloc(sizeof(float) * Matrixsize * Matrixsize);
    clock_t start_host = clock();
    MatrixMul_host(a_host,Matrixsize,Matrixsize,b_host,Matrixsize,Matrixsize,result_cpu);
    cout<<"cputime="<<(double)(clock() - start_host)/1000<<"ms"<<endl;
    // MatrixPrint(result_cpu,Matrixsize,Matrixsize);


//compare value of gpu and cpu
    float err=MatrixCompare(result_cpu,result_host,Matrixsize,Matrixsize);
    cout<<"err in gpu and cpu = "<<err<<endl;

    return 0;
}