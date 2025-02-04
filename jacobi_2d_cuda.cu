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
        cudaDeviceSynchronize();
    }
    stop_cuda_kernel(timer);
	checkCudaError("cudaKernel fault v3");

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
            if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 0)

            // Calcoliamo solo se gli indici sono validi nella memoria condivisa
            if (i_shared_offset+i_shared < WORK_BLOCK &&
                j_shared_offset+j_shared < WORK_BLOCK) {
                if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 0)

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

    // Creazione degli stream
    cudaStream_t *streams = new cudaStream_t[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

   // Allocazione di memoria UVM per ogni stream
    DATA_TYPE **uvm_A = new DATA_TYPE*[NUM_STREAMS];
    DATA_TYPE **uvm_B = new DATA_TYPE*[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMallocManaged(&uvm_A[i], n * n * sizeof(DATA_TYPE));
        cudaMallocManaged(&uvm_B[i], n * n * sizeof(DATA_TYPE));
    }

    // Suddivisione della matrice in blocchi per gli stream
    int rows_per_stream = (n + NUM_STREAMS - 1) / NUM_STREAMS;

    start_cuda_kernel(timer);
    for (int t = 0; t < tsteps; t++) {
      for (int i = 0; i < NUM_STREAMS; i++) {
            int start_row = i * (rows_per_stream - 2 * WORK_BLOCK); // Sovrapposizione per i bordi
            int end_row = (i == NUM_STREAMS - 1) ? n : (i + 1) * (rows_per_stream - 2 * WORK_BLOCK);

            // Aggiungi i bordi
            int start_row_with_borders = max(0, start_row - WORK_BLOCK);
            int end_row_with_borders = min(n, end_row + WORK_BLOCK);

            // Copia asincrona dei dati in UVM per ogni stream
            cudaMemcpyAsync(uvm_A[i] + start_row_with_borders * n,
                            A + start_row_with_borders * n,
                            (end_row_with_borders - start_row_with_borders) * n * sizeof(DATA_TYPE),
                            cudaMemcpyHostToDevice, streams[i]);

            // Esecuzione del kernel per ogni stream
            kernel_jacobi_cuda_v2<<<grid, block, 0, streams[i]>>>(n, uvm_A[i], uvm_B[i]);

            // Sincronizzazione dello stream
            cudaStreamSynchronize(streams[i]);

			cudaMemcpyAsync(A + start_row * n,
                            uvm_A[i] + start_row * n,
                            (end_row - start_row) * n * sizeof(DATA_TYPE),
                            cudaMemcpyDeviceToHost, streams[i]);
	  }

    }
    stop_cuda_kernel(timer);
	checkCudaError("cudaKernel fault v2");

     for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(uvm_A[i]);
        cudaFree(uvm_B[i]);
    }
    delete[] uvm_A;
    delete[] uvm_B;

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

/*
__global__ void kernel_jacobi_cuda_v2(int n, DATA_TYPE *__restrict__ A, DATA_TYPE *__restrict__ B, int offset_i, int offset_j) {
      // Dimensioni del blocco
    int i_start = blockIdx.y * blockDim.y + (threadIdx.y * STREAM_TILE_W)+offset_i;
    int j_start = blockIdx.x * blockDim.x + (threadIdx.x * STREAM_TILE_W) + offset_j;
    printf("i_start %d, j_start %d\n", i_start, j_start);

    // Indici per accedere alla memoria condivisa
    int i_shared = threadIdx.y + 1;  // Offset di 1 per i bordi
    int j_shared = threadIdx.x + 1;  // Offset di 1 per i bordi

    // Allocazione della memoria condivisa (dimensione fissa)
    __shared__ DATA_TYPE A_sub[WORK_BLOCK][WORK_BLOCK];

    int max_i = (offset_i+n/NUM_STREAMS);
    int max_j = (offset_j+n/NUM_STREAMS);
    //usciamo se il numero del thread supera il numero di elementi da calcolare nello stream(altri stream calcoleranno queso elemento)
    if(i_start > max_i || j_start > max_j) {
        return;
    }

    // Carichiamo i dati dalla memoria globale nella memoria condivisa, includendo i bordi

    for (int idx = 0; idx < STREAM_TILE_W * STREAM_TILE_W; idx++) {
        int i = idx / STREAM_TILE_W;
        int j = idx % STREAM_TILE_W;
        int global_i = i + i_start;
        int global_j = j + j_start;
        if (global_i < max_i && global_j < max_j) {
            A_sub[i_shared + i][j_shared + j] = A[global_i * n + global_j];
        }
    }

    // Caricare i bordi (righe e colonne precedenti)
    // Top border
    if (threadIdx.y == 0 && i_start > 0) {
        //printf("Stream=%d Blocco: (%d,%d): caricando in A_sub: A_sub(%d,%d) elemento A(%d)\n", (offset_i*(n/NUM_STREAMS)+offset_j/(n/NUM_STREAMS)), blockIdx.x, blockIdx.y, 0, (j_shared), (i_start - 1) * n + (j_start + threadIdx.x));
        A_sub[0][j_shared] = A[(i_start - 1) * n + (j_start + threadIdx.x)];

    }
    // Left border
    if (threadIdx.x == 0 && j_start > 0) {
        A_sub[i_shared][0] = A[(i_start + threadIdx.y) * n + (j_start - 1)];
    }
    // Bottom border
    if (threadIdx.y == blockDim.y - 1 && i_start + blockDim.y * STREAM_TILE_W < n) {
        A_sub[WORK_BLOCK - 1][j_shared] = A[(i_start + blockDim.y * STREAM_TILE_W) * n + (j_start + threadIdx.x)];
    }
    // Right border
    if (threadIdx.x == blockDim.x - 1 && j_start + blockDim.x * STREAM_TILE_W < n) {
        A_sub[i_shared][WORK_BLOCK - 1] = A[(i_start + threadIdx.y) * n + (j_start + blockDim.x * STREAM_TILE_W)];
    }

    // Sincronizzare i thread per assicurarsi che tutti i dati siano caricati nella memoria condivisa
    __syncthreads();

    // Calcoliamo gli elementi, usando la memoria condivisa se disponibile
    for (int i = i_start; i < i_start + STREAM_TILE_W && i < n - 1; i++) {
      if(i==0) ++i;
        for (int j = j_start; j < j_start + STREAM_TILE_W && j < n - 1; j++) {
          if(j==0) ++j;
            int i_shared_offset = i - (i_start );  // Mappatura dell'indice globale in memoria condivisa
            int j_shared_offset = j - (j_start);  // Mappatura dell'indice globale in memoria condivisa

            // Calcoliamo solo se gli indici sono validi nella memoria condivisa
            //salviamo direttamente in A dato che i dati vengono presi da A_sub per questo passo
            if (i_shared_offset+i_shared < WORK_BLOCK &&
                j_shared_offset+j_shared < WORK_BLOCK) {
                A[i * n + j] = 0.2 * (A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared] +
                                      A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared - 1] +
                                      A_sub[i_shared+i_shared_offset][j_shared_offset+j_shared + 1] +
                                      A_sub[i_shared+i_shared_offset + 1][j_shared_offset+j_shared] +
                                      A_sub[i_shared+i_shared_offset - 1][j_shared_offset+j_shared]);
                }
        }
    }

}

void kernel_jacobi_cuda_v2_host(int tsteps, int n, DATA_TYPE *A, DATA_TYPE *B, Timing *timer) {
    dim3 block(NUM_THREAD_BLOCK, NUM_THREAD_BLOCK);
    int el_per_thread = (NUM_THREAD_BLOCK * TILE_W);
    int size_x = ((n-2)/NUM_STREAMS + block.x - 1); int size_y = ((n-2) + block.y - 1);

    dim3 grid((n/NUM_STREAMS + block.x - 1) / (NUM_THREAD_BLOCK * STREAM_TILE_W), (n/NUM_STREAMS + block.y - 1) / (NUM_THREAD_BLOCK * STREAM_TILE_W));

    DATA_TYPE *uvm_A, *uvm_B;
	// Creazione di stream CUDA (come matrice)
    cudaStream_t streams[NUM_STREAMS*NUM_STREAMS];

    // Allocazione di memoria UVM
    if (cudaMallocManaged(&uvm_A, n * n * sizeof(DATA_TYPE)) != cudaSuccess) {
        printf("Error allocating memory for uvm_A!\n");
    }
    if (cudaMallocManaged(&uvm_B, n * n * sizeof(DATA_TYPE)) != cudaSuccess) {
        printf("Error allocating memory for uvm_B!\n");
    }
    cudaMemcpy(uvm_A, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(uvm_B, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    start_cuda_kernel(timer);

    //Jetson ha 4 thread cpu nel caso in cui facciamo partire solo 4 stream sarà possibile eseguire la parte host di ogni stream
    //con un thread cpu diverso
#pragma omp parallel num_threads(NUM_STREAMS*NUM_STREAMS)
    {
        if (cudaStreamCreate(&streams[omp_get_thread_num()]) != cudaSuccess) {
            printf("Error creating stream %d\n", omp_get_thread_num());
        }
   for (int t = 0; t < tsteps; t++) {

#pragma omp for schedule(static)
        for (int s = 0; s < NUM_STREAMS*NUM_STREAMS; s++) {

                int i= s/NUM_STREAMS;
                int j= s%NUM_STREAMS;
                int offset_i = i * (n / NUM_STREAMS);  // Offset delle righe
                int offset_j = j * (n / NUM_STREAMS);  // Offset delle colonne
                ///*
                parte commentata poichè caricare in memoria dati parziali e far partire il kernel mano a mano
                non sembra aver dato alcun vantaggio rispetto a copia unica iniziale
            	if (t == 0) {
            	    int height = (i == NUM_STREAMS - 1) ? n - offset_i : (n / NUM_STREAMS);
            	    int width = (j == NUM_STREAMS - 1) ? n - offset_j : (n / NUM_STREAMS);


            	    // Copia asincrona della sottomatrice
            	    cudaMemcpy2DAsync(&uvm_A[offset_i * n + offset_j], n * sizeof(DATA_TYPE), &A[offset_i * n + offset_j], n * sizeof(DATA_TYPE),
                        width * sizeof(DATA_TYPE), height, cudaMemcpyHostToDevice,
                        streams[s]);
            	    cudaStreamSynchronize(streams[s]);
                }
                 //--fine_commento/
                // Lancia il kernel sullo stream correlato
                printf("offset_i %d, offset_j %d\n", offset_i, offset_j);
                kernel_jacobi_cuda_v2<<<grid, block, 0, streams[omp_get_thread_num()]>>>(
                    n, uvm_A, uvm_B, offset_i, offset_j);

                //sincronizza stream
                //dobbiamo assicurarci non solo che lo stream per il thread corrente sia uguale, ma anche che i thread precedenti e successivi
                //nella colonna e nella riga sia sincronizzati, questo perchè i dati degli altri stream verranno usati per i bordi all'interno dello
                //stream attuale
                cudaStreamSynchronize(streams[omp_get_thread_num()]);

                // Assicurati che i bordi siano aggiornati (sincronizzazione tra gli stream vicini)
                if (i > 0) cudaStreamSynchronize(streams[(i - 1) * NUM_STREAMS + j]); // Stream sopra
                if (i < NUM_STREAMS - 1) cudaStreamSynchronize(streams[(i + 1) * NUM_STREAMS + j]); // Stream sotto
                if (j > 0) cudaStreamSynchronize(streams[i * NUM_STREAMS + (j - 1)]); // Stream a sinistra
                if (j < NUM_STREAMS - 1) cudaStreamSynchronize(streams[i * NUM_STREAMS + (j + 1)]); // Stream a destra

        }
       //la sincronizzazione dei thread cpu non è gestita appositamente per fine del for ogni thread cpu una volta arrivato qui avrà
        // aspettato lo stream a lui legato e i vicini, di conseguenza il thread corrente ha tutti i dati che gli servono
        // per procedere al prossimo step senza sincronizzare ulteriormente

       //N.B questo è possibile solo perchè NUM_THREADS = NUM_STREAMS*NUM_STREAMS, se cambiassimo i valori saranno da scommentare i controlli
       //precedenti per sincronizzare gli altri stream
        //printf("end step: %d\n", t);
	}
    cudaStreamDestroy(streams[omp_get_thread_num()]);

}
    //sincronizzazione implicita di openMP
	checkCudaError("cudaKernel fault v4");
    //
    stop_cuda_kernel(timer);

    // Copiamo il risultato finale in A
    cudaMemcpy(A, uvm_A, n * n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Deallocazione
    cudaFree(uvm_A);
    cudaFree(uvm_B);/*
    for (int i = 0; i < NUM_STREAMS * NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        checkCudaError("cudaStreamDestroy error");
    }
}
*/

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