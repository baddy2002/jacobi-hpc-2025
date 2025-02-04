#include <stdio.h>
#include <stdlib.h>
#include "timing.h"
#include <math.h>
#include <omp.h>
/* Include benchmark-specific header. */
/* Default data type is double, default size is 20x1000. */
#include "jacobi-2d-imper.h"
#ifdef __CUDA__
#include "jacobi_2d_cuda.cuh"
#endif

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE **A)

{
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);

    }
    fprintf(stderr, "\n");
  }

}


static void print_array_1D(int n,
                        DATA_TYPE *A)

{
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i*n+j]);

    }
    fprintf(stderr, "\n");
  }

}



/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE **A,
                       DATA_TYPE **B)
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */

static void kernel_jacobi_2d_imper(int tsteps,
                                   int n,
                                   DATA_TYPE **A,
                                   DATA_TYPE **B)
{
  int t, i, j;

  for (t = 0; t < tsteps; t++)
  {
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < n - 1; j++)
        B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < n - 1; j++)
        A[i][j] = B[i][j];
  }
}

static void kernel_cpu_jacobi_v1(int tsteps,
                                   int n,
                                   DATA_TYPE **A,
                                   DATA_TYPE **B)
{
  int t, i, j;

  for (t = 0; t < tsteps; t++)
  {
#pragma omp parallel for schedule(static)
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < n - 1; j++)
        B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
#pragma omp parallel for schedule(static)
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < n - 1; j++)
        A[i][j] = B[i][j];

  }
}


static void kernel_cpu_jacobi_v2(int tsteps,
                                   int n,
                                   DATA_TYPE **A,
                                   DATA_TYPE **B)
{
  int t, i, j;

  for (t = 0; t < tsteps; t++)
  {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (i = 1; i < n - 1; i++)
        for (j = 1; j < n - 1; j++)
          B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
#pragma omp barrier
#pragma omp for schedule(static)
      for (i = 1; i < n - 1; i++)
        for (j = 1; j < n - 1; j++)
          A[i][j] = B[i][j];
    } //pragma
  } //for
}


static void kernel_cpu_jacobi_v3(int tsteps,
                                   int n,
                                   DATA_TYPE **A,
                                   DATA_TYPE **B)
{
  int t, i, j;

  for (t = 0; t < tsteps; t++)
  {
#pragma parallel omp for schedule(static)
      for (i = 1; i < n - 1; i++)
#pragma omp simd
        for (j = 1; j < n - 1; j++)
          B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
#pragma omp barrier
#pragma parallel omp for schedule(static)
      for (i = 1; i < n - 1; i++)
#pragma omp simd
        for (j = 1; j < n - 1; j++)
          A[i][j] = B[i][j];
  } //for
}


static void kernel_cpu_jacobi_v4(int tsteps,
                                   int n,
                                   DATA_TYPE **A,
                                   DATA_TYPE **B)
{
  int t, i, j;
  for (t = 0; t < tsteps; t++) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 1; bi < n - 1; bi += BLOCK_SIZE) {
      for (int bj = 1; bj < n - 1; bj += BLOCK_SIZE) {
        for (i = bi; i < MIN(bi + BLOCK_SIZE, n - 1); i++) {
#pragma omp simd
          for (j = bj; j < MIN(bj + BLOCK_SIZE, n - 1); j++) {
            B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
          }
        }
      }
    }

    // Copia B -> A
#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 1; bi < n - 1; bi += BLOCK_SIZE) {
      for (int bj = 1; bj < n - 1; bj += BLOCK_SIZE) {
        for (i = bi; i < MIN(bi + BLOCK_SIZE, n - 1); i++) {
#pragma omp simd
          for (j = bj; j < MIN(bj + BLOCK_SIZE, n - 1); j++) {
            A[i][j] = B[i][j];
          }
        }
      }
    }
  }
}

static void kernel_gpu_jacobi_v1(int tsteps,
                                   int n,
                                   DATA_TYPE *A,
                                   DATA_TYPE *B)
{
  int t, i, j;

  // Gestione dei dati di A e B tra host e device
#pragma omp target enter data map(to:n,A[0:n*n]) map(alloc:B[0:n*n])
  int teams =  (n*n + NTHREADS_GPU - 1)  / NTHREADS_GPU;

  for (t = 0; t < tsteps; t++) {
    // Prima regione target per il calcolo su B
    {
#pragma omp target teams num_teams(teams) thread_limit(NTHREADS_GPU)
#pragma omp distribute parallel for collapse(2) num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU) schedule(static, 1)
      for (i = 1; i < n - 1; i++) {
        for (j = 1; j < n - 1; j++) {
          B[i * n + j] = 0.2 * (A[i * n + j] +
                                 A[i * n + (j-1)] +
                                 A[i * n + (j+1)] +
                                 A[(i+1) * n + j] +
                                 A[(i-1) * n + j]);
        }
      }
    }

    // Seconda regione target per il calcolo su A
#pragma omp target teams num_teams(teams) thread_limit(NTHREADS_GPU)
#pragma omp distribute parallel for collapse(2) num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU) schedule(static, 1)
  for (i = 1; i < n - 1; i++)
    for (j = 1; j < n - 1; j++)
          A[i * n + j] = B[i * n + j];

  }

  // Uscita dei dati dalla memoria del device
#pragma omp target exit data map(from: A[0:n*n]) map(release: B)
}



int main(int argc, char **argv)
{
  #ifdef _OPENMP
    printf("OpenMP è abilitato. Versione: %d\n", _OPENMP);
#endif

#ifdef __CUDA__
    printf("CUDA è abilitato. ");
    check_device_properties();
#endif
#ifdef GPU
  printf("Il programma eseguirà su GPU!");
#else
  printf("Il programma eseguirà su CPU!");
#endif


  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  DATA_TYPE** A = malloc(n * sizeof(DATA_TYPE*));
  DATA_TYPE** B = malloc(n * sizeof(DATA_TYPE*));
  for (int i = 0; i < n; i++) {
    A[i] = (DATA_TYPE*)malloc(n * sizeof(DATA_TYPE));
    B[i] = (DATA_TYPE*)malloc(n * sizeof(DATA_TYPE));
  }

  /* Initialize array(s). */
  init_array(n, A, B);
#ifdef GPU
  DATA_TYPE *A_1D = (DATA_TYPE *)malloc(n*n * sizeof(DATA_TYPE));
  DATA_TYPE *B_1D = (DATA_TYPE *)malloc(n*n * sizeof(DATA_TYPE));
printf("initializing...");
  // Copy 2D array to 1D for better memory handling
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      A_1D[i * n + j] = A[i][j];
      B_1D[i * n + j] = B[i][j];
    }
  }
  #endif

  /* Timer */
  Timing timer;


  printf("Inizio esecuzione...\n");


  /* Run kernel. */
#ifdef CPU_V1

  start_instruments(&timer, 0); // CPU
  kernel_cpu_jacobi_v1(tsteps, n, A, B);
  stop_instruments(&timer, 0);
#elif defined(CPU_V2)
  start_instruments(&timer, 0); // CPU
  kernel_cpu_jacobi_v2(tsteps, n, A, B);
  stop_instruments(&timer, 0);
#elif defined(CPU_V3)
  start_instruments(&timer, 0); // CPU
  kernel_cpu_jacobi_v3(tsteps, n, A, B);
  stop_instruments(&timer, 0);
#elif defined(CPU_V4)
  start_instruments(&timer, 0); // CPU
  kernel_cpu_jacobi_v4(tsteps, n, A, B);
  stop_instruments(&timer, 0);
#elif defined(GPU_V1)
  start_instruments(&timer, 1); // GPU
  kernel_gpu_jacobi_v1(tsteps, n, A_1D, B_1D);
  stop_instruments(&timer, 1);
#elif defined(CUDA_V1)
  start_instruments(&timer, 1); // GPU
  kernel_jacobi_cuda_v1_host(tsteps, n, A_1D, B_1D, &timer);
  stop_instruments(&timer, 1);
#elif defined(CUDA_V2)
  start_instruments(&timer, 1); // GPU
  kernel_jacobi_cuda_v2_host(tsteps, n, A_1D, B_1D, &timer);
  stop_instruments(&timer, 1);
#else
  start_instruments(&timer, 0); // CPU di default
  kernel_jacobi_2d_imper(tsteps, n, A, B);
  stop_instruments(&timer, 0);
#endif

  printf("Fine esecuzione\n");

  /* Stampa dei risultati del timer */
#ifdef GPU
  print_instruments(&timer, 1);
#elif defined(__CUDA__)
  print_instruments(&timer, 1);
#else
  print_instruments(&timer, 0);
#endif
#ifdef GPU
  // Copy 1D array to 2D for print
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
     A[i][j] =  A_1D[i * n + j];
    }
  }

#endif

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  /* Stampa l'array risultante */
  print_array(n, A);

  /* clean */
  for (int i = 0; i < n; i++) {
    free(A[i]);
    free(B[i]);
  }
#ifdef GPU
  free(A_1D);
  free(B_1D);
#endif
  free(A);
  free(B);

  return 0;


}
