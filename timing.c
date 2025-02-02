#include "timing.h"

void start_instruments(Timing* timer, int use_gpu) {
    if (use_gpu) {
#ifdef __CUDA__
        cudaEventCreate(&timer->gpu_start);
        cudaEventCreate(&timer->gpu_stop);
        cudaEventCreate(&timer->mem_start);
        cudaEventCreate(&timer->mem_stop);
        cudaEventRecord(timer->mem_start, 0);
#elif defined(_OPENMP)
        timer->omp_start = omp_get_wtime();
#endif
    } else {
        timer->cpu_start = clock();
    }
}

void start_cuda_kernel(Timing* timer) {
    #ifdef __CUDA__
    cudaEventRecord(timer->gpu_start, 0);
    #endif
}

void stop_cuda_kernel(Timing* timer) {
#ifdef __CUDA__
    cudaEventRecord(timer->gpu_stop, 0);
    cudaEventSynchronize(timer->gpu_stop);
    cudaEventElapsedTime(&timer->elapsed_gpu_time, timer->gpu_start, timer->gpu_stop);
#endif
}

void stop_instruments(Timing* timer, int use_gpu) {
    if (use_gpu) {
#ifdef __CUDA__
        cudaEventRecord(timer->mem_stop, 0);
        cudaEventSynchronize(timer->mem_stop);
        cudaEventElapsedTime(&timer->elapsed_mem_time, timer->mem_start, timer->mem_stop);
        cudaEventDestroy(timer->gpu_start);
        cudaEventDestroy(timer->gpu_stop);
        cudaEventDestroy(timer->mem_start);
        cudaEventDestroy(timer->mem_stop);
        timer->elapsed_mem_time = timer->elapsed_mem_time - timer->elapsed_gpu_time;
#elif defined(_OPENMP)
        timer->omp_end = omp_get_wtime();
        timer->elapsed_omp_time = timer->omp_end - timer->omp_start;
#endif
    } else {
        timer->cpu_end = clock();
        timer->elapsed_cpu_time = (double)(timer->cpu_end - timer->cpu_start) / CLOCKS_PER_SEC;
    }
}

void print_instruments(Timing* timer, int use_gpu) {
    if (use_gpu) {
#ifdef __CUDA__
        printf("Tempo GPU Kernel con CUDA: %f ms\n", timer->elapsed_gpu_time);
        printf("Tempo copia mem su GPU: %f ms\n", timer->elapsed_mem_time);
#elif defined(_OPENMP)
        printf("Tempo GPU con OpenMP: %f s\n", timer->elapsed_omp_time);
#endif
    } else {
        printf("Tempo CPU: %f s\n", timer->elapsed_cpu_time);
    }
}
