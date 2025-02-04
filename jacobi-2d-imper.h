#ifndef JACOBI_2D_IMPER_H
# define JACOBI_2D_IMPER_H

#ifdef __cplusplus
extern "C" {
#endif
/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(TSTEPS) && ! defined(N)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define TSTEPS 100
#   define N 128
#  endif

#  ifdef SMALL_DATASET
#   define TSTEPS 500
#   define N 512
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define TSTEPS 1000
#   define N 1024
#  endif

#  ifdef LARGE_DATASET
#   define TSTEPS 2000
#   define N 2048
#  endif

#  ifdef EXTRALARGE_DATASET
#   define TSTEPS 4000
#   define N 4096
#  endif
# endif /* !N */

# ifndef DATA_TYPE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define NUM_THREADS 4
#define NTHREADS_GPU 256
#define BLOCK_SIZE 4096
#ifdef __cplusplus
}
#endif
#endif /* !JACOBI_2D_IMPER */
