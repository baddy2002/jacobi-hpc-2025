#include <cuda_runtime.h>
#include <stdio.h>
#include "jacobi_2d_cuda.cuh"
#include <omp.h>

// Define DATA_TYPE

#include "jacobi-2d-imper.h"
#include "timing.h"

//____________________________error handling

// Funzione per controllare e gestire gli errori CUDA
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);  // Esci dal programma in caso di errore, o gestisci l'errore come preferisci
    }
}

__global__ void kernel_jacobi_cuda_v1(int n, DATA_TYPE *A, DATA_TYPE *B) {
      // Indici globale
    int i_start = blockIdx.y * blockDim.y + (threadIdx.y * TILE_W);
    int j_start = blockIdx.x * blockDim.x + (threadIdx.x * TILE_W);

    // Indici per accedere alla memoria condivisa
    int i_shared = threadIdx.y*TILE_W+1;  // +1 per bordi
    int j_shared = threadIdx.x*TILE_W+1;

    // Allocazione della memoria condivisa (dimensione fissa)
    __shared__ DATA_TYPE A_sub[WORK_BLOCK][WORK_BLOCK];

    //dobbiamo caricare TILE_W elementi per thread (quelli da calcolare)
    for (int i = 0; i < TILE_W; i++) {
      int gi = i_start + 1 + i;  //indici globali
        for (int j = 0; j < TILE_W; j++) {
            int gj = j_start + 1 + j;

            if (gi < n && gj < n) {  // Controllo bounds
              //carica elementi che deve calcolare
              A_sub[i_shared + i][j_shared + j] = A[gi * n + gj];

              //se il thread è nella prima riga del blocco deve caricare anche l'elemento del bordo alto
			  	if(threadIdx.y == 0){
					A_sub[0][j_shared + j] = A[(gi-1) * n + gj];
              	}
              //se il thread è nell'ultima riga del blocco deve caricare anche l'elemento del bordo basso
				if (threadIdx.y == blockDim.y - 1) {
					A_sub[WORK_BLOCK - 1][j_shared + j] = A[(gi+1) * n + gj];
        	  	}
              //se il thread è nella prima colnna del blocco deve caricare anche l'elemento del bordo a sinistra
			   if (threadIdx.x == 0) {
					A_sub[i_shared + i][0] = A[gi * n + gj-1];
        		}
              //se il thread è nell'ultima colonna del blocco deve caricare anche l'elemento del bordo a destra
				if (threadIdx.x == blockDim.x - 1) {
                    A_sub[i_shared + i][WORK_BLOCK - 1] = A[gi * n + gj+1];
        		}
            }
        }
    }
    // Sincronizzare i thread per assicurarsi che tutti i dati siano caricati nella memoria condivisa
    __syncthreads();

    // Calcoliamo gli elementi, usando la memoria condivisa se disponibile
    for (int i = i_start+1; i < i_start + TILE_W +1 && i < n - 1; i++) {
        for (int j = j_start+1; j < j_start + TILE_W+1 && j < n - 1; j++) {
            int i_shared_offset = i - (i_start+1 );  // nel caso di TILE_W>0
            int j_shared_offset = j - (j_start+1);
            // Calcoliamo solo se gli indici sono validi nella memoria condivisa
            if (i_shared_offset+i_shared < WORK_BLOCK &&
                j_shared_offset+j_shared < WORK_BLOCK) {

                B[i * n + j] = 0.2 * (A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared] +
                                      A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared - 1] +
                                      A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared + 1] +
                                      A_sub[i_shared+i_shared_offset + 1][j_shared_offset+j_shared] +
                                      A_sub[i_shared+i_shared_offset - 1][j_shared_offset+j_shared]);
                }
        }
    }

     for (int i = i_start+1; i < i_start + TILE_W+1 && i < n - 1; i++) {
        for (int j = j_start+1; j < j_start + TILE_W+1 && j < n - 1; j++) {
            A[i * n + j] = B[i * n + j];
    	}
    }

}

void kernel_jacobi_cuda_v1_host(int tsteps, int n, DATA_TYPE *A, DATA_TYPE *B, Timing *timer) {
    dim3 block(NUM_THREAD_BLOCK, NUM_THREAD_BLOCK);
    int el_per_thread = (NUM_THREAD_BLOCK * TILE_W);
    int size_x = ((n-2)+ block.x - 1); int size_y = ((n-2) + block.y - 1);
    dim3 grid( size_x / el_per_thread, size_y / el_per_thread );

    DATA_TYPE *uvm_A, *uvm_B;

    // Allocazione di memoria UVM
    cudaMallocManaged(&uvm_A, n * n * sizeof(DATA_TYPE));
    cudaMallocManaged(&uvm_B, n * n * sizeof(DATA_TYPE));

    // Copiamo i dati iniziali in UVM
    cudaMemcpy(uvm_A, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(uvm_B, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    start_cuda_kernel(timer);
    for (int t = 0; t < tsteps; t++) {
        kernel_jacobi_cuda_v1<<<grid, block>>>(n, uvm_A, uvm_B);
        //non serve sincronizzazione perchè lanciare kernel su stesso stream garantisce esecuzione in ordine seriale
        //cudaDeviceSynchronize();
    }
    stop_cuda_kernel(timer);
	checkCudaError("cudaKernel fault v1");

    // Copiamo il risultato finale in A
    cudaMemcpy(A, uvm_A, n * n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    // Deallocazione
    cudaFree(uvm_A);
    cudaFree(uvm_B);
}

__global__ void kernel_jacobi_cuda_v2(int n, DATA_TYPE *A, DATA_TYPE *B) {
      // Indici globale
    int i_start = blockIdx.y * blockDim.y + (threadIdx.y * TILE_W);
    int j_start = blockIdx.x * blockDim.x + (threadIdx.x * TILE_W);

    // Indici per accedere alla memoria condivisa
    int i_shared = threadIdx.y*TILE_W+1;  // +1 per bordi
    int j_shared = threadIdx.x*TILE_W+1;

    // Allocazione della memoria condivisa (dimensione fissa)
    __shared__ DATA_TYPE A_sub[WORK_BLOCK][WORK_BLOCK];

    //dobbiamo caricare TILE_W elementi per thread (quelli da calcolare)
    for (int i = 0; i < TILE_W; i++) {
      int gi = i_start + 1 + i;  //indici globali
        for (int j = 0; j < TILE_W; j++) {
            int gj = j_start + 1 + j;

            if (gi < n && gj < n) {  // Controllo bounds
              //carica elementi che deve calcolare
              A_sub[i_shared + i][j_shared + j] = A[gi * n + gj];

              //se il thread è nella prima riga del blocco deve caricare anche l'elemento del bordo alto
			  	if(threadIdx.y == 0){
					A_sub[0][j_shared + j] = A[(gi-1) * n + gj];
              	}
              //se il thread è nell'ultima riga del blocco deve caricare anche l'elemento del bordo basso
				if (threadIdx.y == blockDim.y - 1) {
					A_sub[WORK_BLOCK - 1][j_shared + j] = A[(gi+1) * n + gj];
        	  	}
              //se il thread è nella prima colnna del blocco deve caricare anche l'elemento del bordo a sinistra
			   if (threadIdx.x == 0) {
					A_sub[i_shared + i][0] = A[gi * n + gj-1];
        		}
              //se il thread è nell'ultima colonna del blocco deve caricare anche l'elemento del bordo a destra
				if (threadIdx.x == blockDim.x - 1) {
                    A_sub[i_shared + i][WORK_BLOCK - 1] = A[gi * n + gj+1];
        		}
            }
        }
    }
    // Sincronizzare i thread per assicurarsi che tutti i dati siano caricati nella memoria condivisa
    __syncthreads();
    // Calcoliamo gli elementi, usando la memoria condivisa se disponibile
    for (int i = i_start+1; i < i_start + TILE_W +1 && i < n - 1; i++) {
        for (int j = j_start+1; j < j_start + TILE_W+1 && j < n - 1; j++) {
            int i_shared_offset = i - (i_start+1 );  // nel caso di TILE_W>0
            int j_shared_offset = j - (j_start+1);

            // Calcoliamo solo se gli indici sono validi nella memoria condivisa
            if (i_shared_offset+i_shared < WORK_BLOCK &&
                j_shared_offset+j_shared < WORK_BLOCK) {

                B[i * n + j] = 0.2 * (A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared] +
                                      A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared - 1] +
                                      A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared + 1] +
                                      A_sub[i_shared+i_shared_offset + 1][j_shared_offset+j_shared] +
                                      A_sub[i_shared+i_shared_offset - 1][j_shared_offset+j_shared]);
                }
        }
    }

     for (int i = i_start+1; i < i_start + TILE_W+1 && i < n - 1; i++) {
        for (int j = j_start+1; j < j_start + TILE_W+1 && j < n - 1; j++) {
            A[i * n + j] = B[i * n + j];
    	}
    }
}
void kernel_jacobi_cuda_v2_host(int tsteps, int n, DATA_TYPE *A, DATA_TYPE *B, Timing *timer) {
    dim3 block(NUM_THREAD_BLOCK, NUM_THREAD_BLOCK);
    int el_per_thread = (NUM_THREAD_BLOCK * TILE_W);
    int size_x = ((n-2)+ block.x - 1); int size_y = ((n-2) + block.y - 1);
    dim3 grid( size_x / el_per_thread, size_y / el_per_thread );
    #ifndef ROW_DEEP
	#define ROW_DEEP ((NUM_STREAMS > (grid.y)) ? (NUM_STREAMS) : (grid.y))
	#endif
    // Creazione degli stream
    cudaStream_t *streams = new cudaStream_t[ROW_DEEP];
    for (int i = 0; i < ROW_DEEP; i++) {
        cudaStreamCreate(&streams[i]);
    }

    DATA_TYPE *uvm_A, *uvm_B;

    // Allocazione di memoria UVM
    cudaMallocManaged(&uvm_A, n * n * sizeof(DATA_TYPE));
    cudaMallocManaged(&uvm_B, n * n * sizeof(DATA_TYPE));

    start_cuda_kernel(timer);

#pragma omp parallel num_threads(ROW_DEEP)
{
    for (int t = 0; t < tsteps; t++) {
      int thread_num = omp_get_thread_num();
      //un thread per stream, ogni thread aspetterà il rispettivo stream
#pragma omp for
      for (int i = 0; i < ROW_DEEP; i++) {
        	//se la dimensione della griglia è maggiore degli stream, uno stesso stream eseguirà più chiamate
			int start = (thread_num)*(WORK_BLOCK-1)-1*(thread_num);
            // Copia asincrona dei dati in UVM per ogni stream ()
            if(t==0){
                        cudaMemcpyAsync(&uvm_A[start * n],
                &A[start * n],
                WORK_BLOCK * n * sizeof(DATA_TYPE),
                cudaMemcpyHostToDevice, streams[thread_num]);
			 cudaStreamSynchronize(streams[thread_num]);
             }
            // Esecuzione del kernel per ogni stream
            kernel_jacobi_cuda_v2<<<grid, block, 0, streams[thread_num]>>>(n, uvm_A, uvm_B);

			cudaMemcpyAsync(&A[(start+1)*n],
                            &uvm_A[(start+1)*n],
                            (WORK_BLOCK-2) * n * sizeof(DATA_TYPE),
                            cudaMemcpyDeviceToHost, streams[thread_num]);
            //aspetta copia thread corrente
			cudaStreamSynchronize(streams[thread_num]);
	  }

        if(thread_num > 0)
              cudaStreamSynchronize(streams[thread_num-1]);			//ha calcolato il bordo sopra del prossimo step
        if(thread_num < omp_get_num_threads()-1)
              cudaStreamSynchronize(streams[thread_num+1]);
    }
}
    stop_cuda_kernel(timer);
	checkCudaError("cudaKernel fault v2");

    cudaFree(uvm_A);
    cudaFree(uvm_B);

    for (int i = 0; i < ROW_DEEP; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

__global__ void kernel_jacobi_cuda_v3(int n, DATA_TYPE *A, DATA_TYPE *B) {
      // Indici globale
    int i_start = blockIdx.y * blockDim.y + (threadIdx.y * TILE_W);
    int j_start = blockIdx.x * blockDim.x + (threadIdx.x * TILE_W);

    // Indici per accedere alla memoria condivisa
    int i_shared = threadIdx.y*TILE_W+1;  // +1 per bordi
    int j_shared = threadIdx.x*TILE_W+1;

    // Allocazione della memoria condivisa (dimensione fissa)
    __shared__ DATA_TYPE A_sub[WORK_BLOCK][WORK_BLOCK];

    //dobbiamo caricare TILE_W elementi per thread (quelli da calcolare)
    for (int i = 0; i < TILE_W; i++) {
      int gi = i_start + 1 + i;  //indici globali
        for (int j = 0; j < TILE_W; j++) {
            int gj = j_start + 1 + j;

            if (gi < n && gj < n) {  // Controllo bounds
              //carica elementi che deve calcolare
              A_sub[i_shared + i][j_shared + j] = A[gi * n + gj];

              //se il thread è nella prima riga del blocco deve caricare anche l'elemento del bordo alto
			  	if(threadIdx.y == 0){
					A_sub[0][j_shared + j] = A[(gi-1) * n + gj];
              	}
              //se il thread è nell'ultima riga del blocco deve caricare anche l'elemento del bordo basso
				if (threadIdx.y == blockDim.y - 1) {
					A_sub[WORK_BLOCK - 1][j_shared + j] = A[(gi+1) * n + gj];
        	  	}
              //se il thread è nella prima colnna del blocco deve caricare anche l'elemento del bordo a sinistra
			   if (threadIdx.x == 0) {
					A_sub[i_shared + i][0] = A[gi * n + gj-1];
        		}
              //se il thread è nell'ultima colonna del blocco deve caricare anche l'elemento del bordo a destra
				if (threadIdx.x == blockDim.x - 1) {
                    A_sub[i_shared + i][WORK_BLOCK - 1] = A[gi * n + gj+1];
        		}
            }
        }
    }
    // Sincronizzare i thread per assicurarsi che tutti i dati siano caricati nella memoria condivisa
    __syncthreads();

    // Calcoliamo gli elementi, usando la memoria condivisa se disponibile
    for (int i = i_start+1; i < i_start + TILE_W +1 && i < n - 1; i++) {
        for (int j = j_start+1; j < j_start + TILE_W+1 && j < n - 1; j++) {
            int i_shared_offset = i - (i_start+1 );  // nel caso di TILE_W>0
            int j_shared_offset = j - (j_start+1);
            // Calcoliamo solo se gli indici sono validi nella memoria condivisa
            if (i_shared_offset+i_shared < WORK_BLOCK &&
                j_shared_offset+j_shared < WORK_BLOCK) {

                B[i * n + j] = 0.2 * (A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared] +
                                      A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared - 1] +
                                      A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared + 1] +
                                      A_sub[i_shared+i_shared_offset + 1][j_shared_offset+j_shared] +
                                      A_sub[i_shared+i_shared_offset - 1][j_shared_offset+j_shared]);
                }
        }
    }

     for (int i = i_start+1; i < i_start + TILE_W+1 && i < n - 1; i++) {
        for (int j = j_start+1; j < j_start + TILE_W+1 && j < n - 1; j++) {
            A[i * n + j] = B[i * n + j];
    	}
    }

}

void kernel_jacobi_cuda_v3_host(int tsteps, int n, DATA_TYPE *A, DATA_TYPE *B, Timing *timer) {
    dim3 block(NUM_THREAD_BLOCK, NUM_THREAD_BLOCK);
    int el_per_thread = (NUM_THREAD_BLOCK * TILE_W);
    int size_x = ((n-2)+ block.x - 1); int size_y = ((n-2) + block.y - 1);
    dim3 grid( size_x / el_per_thread, size_y / el_per_thread );

    DATA_TYPE *uvm_A, *uvm_B;

    // Allocazione di memoria UVM
    cudaMallocManaged(&uvm_A, n * n * sizeof(DATA_TYPE));
    cudaMallocManaged(&uvm_B, n * n * sizeof(DATA_TYPE));

    // Copiamo i dati iniziali in UVM
    cudaMemcpy(uvm_A, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(uvm_B, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

/*
    // Ora puoi chiamare cudaMemPrefetchAsync con il dispositivo corretto
    cudaMemPrefetchAsync(uvm_A, n * n * sizeof(DATA_TYPE), 0, 0);
    cudaMemPrefetchAsync(uvm_B, n * n * sizeof(DATA_TYPE), 0, 0);
*/

    start_cuda_kernel(timer);
    for (int t = 0; t < tsteps; t++) {
        kernel_jacobi_cuda_v3<<<grid, block>>>(n, uvm_A, uvm_B);
        //non serve sincronizzazione perchè lanciare kernel su stesso stream garantisce esecuzione in ordine seriale
        //cudaDeviceSynchronize();
    }
    stop_cuda_kernel(timer);
	checkCudaError("cudaKernel fault v1");

    // Copiamo il risultato finale in A
    cudaMemcpy(A, uvm_A, n * n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    // Deallocazione
    cudaFree(uvm_A);
    cudaFree(uvm_B);
}

void check_device_properties() {
    int deviceCount = 0;

    // Ottieni il numero di dispositivi GPU disponibili
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    // Cicla su tutti i dispositivi
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("\nDevice %d: %s\n", device, deviceProp.name);
        printf("  Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
        printf("  Total Global Memory (MB): %zu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));  // Corretto
        printf("  Shared Memory per Block (bytes): %zu\n", deviceProp.sharedMemPerBlock);  // Corretto
        printf("  Total Constant Memory (bytes): %zu\n", deviceProp.totalConstMem);  // Corretto

        int l2CacheSize = 0;
        cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, device);
        printf("  L2 Cache Size (bytes): %d\n", l2CacheSize);  // Corretto

        int sharedMemPerSM = 0;
        cudaDeviceGetAttribute(&sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
        printf("  Shared Memory per Multiprocessor (bytes): %d\n", sharedMemPerSM);
        printf("  ManagedAccess: %d\n", deviceProp.concurrentManagedAccess);  //se è 0 non può avere memPrefetch
        printf("  Multiprocessor Count: %d\n", deviceProp.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Shared Memory per Block (bytes): %zu\n", deviceProp.sharedMemPerBlock);  // Corretto
        printf("  Total Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp Size: %d\n", deviceProp.warpSize);
        printf("  Clock Rate (Hz): %d\n", deviceProp.clockRate);
        printf("  Max Grid Size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Max Threads Dim: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    }
}