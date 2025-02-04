BENCHMARK = $(shell basename `pwd`)
SRC = $(BENCHMARK).c timing.c
HEADERS = $(BENCHMARK).h timing.h
CUFILES = jacobi_2d_cuda.cu

DEPS        := Makefile.dep
DEP_FLAG    := -MM

CC = gcc
CLANG = clang
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS=  $(NVOPT)
NVCCLDFLAGS = -lcuda -lcudart
NVOPT= -Xcompiler $(OPT) --ptxas-options=-v --use_fast_math -arch=sm_35
LD = ld
OBJDUMP = objdump
OPT = -fopenmp -O3 -g
CFLAGS = -I. $(OPT) $(EXT_CFLAGS)
LDFLAGS = -lm $(EXT_LDFLAGS)

OMP= -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -DGPU
CUDA = -DGPU -D__CUDA__
# Definisci i target per ciascun kernel
EXE = $(BENCHMARK)_cpu_v0
EXE_CPU_V1 = $(BENCHMARK)_cpu_v1_out
EXE_CPU_V2 = $(BENCHMARK)_cpu_v2_out
EXE_CPU_V3 = $(BENCHMARK)_cpu_v3_out
EXE_CPU_V4 = $(BENCHMARK)_cpu_v4_out
EXE_CPU_V5 = $(BENCHMARK)_cpu_v5_out

EXE_GPU_V1 = $(BENCHMARK)_gpu_v1_out
EXE_GPU_V2 = $(BENCHMARK)_gpu_v2_out
EXE_GPU_V3 = $(BENCHMARK)_gpu_v3_out
EXE_GPU_V4 = $(BENCHMARK)_gpu_v4_out
EXE_GPU_V5 = $(BENCHMARK)_gpu_v5_out

EXE_CUDA_V1 = $(BENCHMARK)_cuda_v1_out
EXE_CUDA_V2 = $(BENCHMARK)_cuda_v2_out
EXE_CUDA_V3 = $(BENCHMARK)_cuda_v3_out
EXE_CUDA_V4 = $(BENCHMARK)_cuda_v4_out


.PHONY: all exe clean veryclean run

# Target per compilare tutti gli eseguibili
all: $(EXE_GPU_V1).o timing.o $(EXE) $(EXE_CPU_V1) $(EXE_CPU_V2) $(EXE_CPU_V3) $(EXE_CPU_V4) $(EXE_GPU_V1) $(EXE_CUDA_V1)  $(EXE_CUDA_V2)

cpu_v1: $(EXE_CPU_V1)
cpu_v2: $(EXE_CPU_V2)
cpu_v3: $(EXE_CPU_V3)
cpu_v4: $(EXE_CPU_V4)

gpu_v1: $(EXE_GPU_V1)

cuda_v1: $(EXE_CUDA_V1)
cuda_v2: $(EXE_CUDA_V2)


$(EXE): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(INCPATHS) $^ -o $@ $(LDFLAGS)

# Esegui la compilazione per ogni versione del kernel
$(EXE_CPU_V1): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(INCPATHS) -DCPU_V1 $^ -o $@ $(LDFLAGS)

$(EXE_CPU_V2): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(INCPATHS) -DCPU_V2 $^ -o $@ $(LDFLAGS)

$(EXE_CPU_V3): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(INCPATHS) -DCPU_V3 $^ -o $@ $(LDFLAGS)

$(EXE_CPU_V4): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(INCPATHS) -DCPU_V4 $^ -o $@ $(LDFLAGS)

timing.o: timing.c timing.h
	$(CLANG) -c $(OMP) $(CFLAGS) timing.c -o timing.o

$(EXE_GPU_V1).o: $(BENCHMARK).c
	$(CLANG) -c $(CFLAGS) $(OMP) $(INCPATHS) -DGPU_V1 $^ -o $@

$(EXE_GPU_V1): $(EXE_GPU_V1).o timing.o
	$(CLANG) $(CFLAGS) $(OMP) -DGPU_V1 $(EXE_GPU_V1).o timing.o -o $@ $(LDFLAGS)

$(EXE_CUDA_V1): $(SRC) $(HEADERS) $(CUFILES)
	$(NVCC) $(NVCCFLAGS) $(CUDA) --ptxas-options=-v -DCUDA_V1 $(SRC) $(CUFILES) -o $@ $(NVCCLDFLAGS)

$(EXE_CUDA_V2): $(SRC) $(HEADERS) $(CUFILES)
	$(NVCC) $(NVCCFLAGS) $(CUDA) --ptxas-options=-v -g -DCUDA_V2 $(SRC) $(CUFILES) -o $@ $(NVCCLDFLAGS)

# Target per pulire i file generati
clean:
	-rm -vf $(EXE) $(EXE_CPU_V1) $(EXE_CPU_V2) $(EXE_CPU_V3) $(EXE_CPU_V4) $(EXE_CPU_V5) timing.o $(EXE_GPU_V1).o $(EXE_GPU_V1) $(EXE_CUDA_V1) $(EXE_CUDA_V2)  *~

# Target per una pulizia completa
veryclean: clean
	-rm -vf $(DEPS)

exe: $(EXE)


run: $(EXE)
	./$(EXE)

# Genera le dipendenze
$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)
