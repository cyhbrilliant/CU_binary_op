#include <iostream>
#include <cuda_runtime.h>
using namespace std;
int get_GPU_Rate()
{
	cudaDeviceProp deviceProp;
 	cudaGetDeviceProperties(&deviceProp,0);
 	return deviceProp.clockRate;
}

__global__ void Xor(int a,int b,int *result_device,clock_t* time){
	clock_t start = clock();
	int c;
	*result_device+=a^b;

	*time = clock() - start;
	
}


int main(){
	int *result_device;
    cudaMalloc((void**) &result_device, sizeof(int));
    clock_t* time;
    cudaMalloc((void**) &time, sizeof(clock_t));

    cudaEvent_t start_device, stop_device;
    float time_device;
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);
    cudaEventRecord( start_device, 0 );

    Xor<<<1,1>>>(1,-1,result_device,time);

	cudaEventRecord( stop_device, 0 );
    cudaEventSynchronize( stop_device );
    cudaEventElapsedTime( &time_device, start_device, stop_device );
    cudaEventDestroy( start_device );
    cudaEventDestroy( stop_device );
    cout<<"gputime="<<time_device<<"ms"<<endl;


	clock_t time_used;
    cudaMemcpy(&time_used, time, sizeof(clock_t),cudaMemcpyDeviceToHost);
    cout<<"time="<<time_used<<endl;

    int result_host;
    cudaMemcpy(&result_host, result_device, sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(result_device);
    cout<<result_host<<endl;

    return 0;

}

