#ifndef TIMING_H
#define TIMING_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __CUDA__
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    clock_t cpu_start, cpu_end;
    double elapsed_cpu_time;
#ifdef _OPENMP
    double omp_start, omp_end, elapsed_omp_time;
#endif
#ifdef __CUDA__
    cudaEvent_t mem_start, mem_stop;
    cudaEvent_t gpu_start, gpu_stop;
    float elapsed_gpu_time, elapsed_mem_time;
#endif
} Timing;

void start_instruments(Timing* timer, int use_gpu);
void stop_instruments(Timing* timer, int use_gpu);
void print_instruments(Timing* timer, int use_gpu);
void start_cuda_kernel(Timing* timer);
void stop_cuda_kernel(Timing* timer);
#ifdef __cplusplus
}
#endif
#endif // TIMING_H
