# Jacobi Project

Questo progetto è un esempio di come creare accelerazioni del metodo di jacobi per OpenMP e Cuda 


Vedi altri miei progetti relativi a calcolo parallelo:
- [Cuda](https://Baddy2002.github.io/Cuda) - Progetti di calcolo parallelo.



Sono stati usati 5 differenti dataset, ognuno di loro è riproducibile indicando nel momento di compilazione con make quello desiderato: 

- -DMINI_DATASET        (N=128, steps=100)
- -DSMALL_DATASET       (N=512, steps=500)
- -DSTANDARD_DATASET    (N=1024, steps=1000)
- -DLARGE_DATASET       (N=2048, steps=2000)
- -DEXTRALARGE_DATASET  (N=4096, steps=4000)

e.g make all EXT_CFLAGS="-DMINI_DATASET" -> compilerà tutti i programmi (tutte le versioni) con un dataset 128*128 e 100 iterazioni
