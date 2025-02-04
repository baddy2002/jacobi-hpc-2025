#ifndef JACOBI_2D_CUDA_H
#define JACOBI_2D_CUDA_H

#include "timing.h"
#include "jacobi-2d-imper.h"

#ifdef __cplusplus
extern "C" {
#endif
    void check_device_properties();
    void kernel_jacobi_cuda_v1_host(int tsteps, int n, DATA_TYPE *A, DATA_TYPE *B, Timing *timer);
    void kernel_jacobi_cuda_v2_host(int tsteps, int n, DATA_TYPE *A, DATA_TYPE *B, Timing *timer);
    void kernel_jacobi_cuda_v3_host(int tsteps, int n, DATA_TYPE *A, DATA_TYPE *B, Timing *timer);

#ifdef __cplusplus
}
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 4
#endif

#ifndef NUM_THREAD_BLOCK
#define NUM_THREAD_BLOCK 16
#endif

#ifndef TILE_W
#define TILE_W ((N / NUM_THREAD_BLOCK) > 65536 ? N / 65536 : 1)
#endif


//un work block è uguale alla dimensione in thread del blocco gpu * il numero di elementi che ogni thread deve calcolare, e gli viene poi aggiunto 2
//in pratica il work_block conterrà tutti gli elementi della matrice da calcolare, in più gli verranno caricate due righe e due colonne in più per permettere
//il calcolo degli elementi ai bordi che altrimenti dovrebbero prendere i dati dei loro vicini dalla global memory
#ifndef WORK_BLOCK
#define WORK_BLOCK (NUM_THREAD_BLOCK*TILE_W + 2)
#endif


#ifndef STREAM_TILE_W
#define STREAM_TILE_W ((N / NUM_THREAD_BLOCK*NUM_STREAMS) > 65536 ? N / 65536 : 1)
#endif


#endif // JACOBI_2D_CUDA_H
