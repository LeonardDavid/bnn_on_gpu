# bnn_on_gpu

1. Navigate to folder of respective model
2. Change desired batch size in ```utils.h```
3. A. run using 
```
$ bash run.bash
```
3. B. alternitavely: set the path to the CUDA Runtime Library of your system in ```Makefile``` 
- currently commented out: the path on my personal PC on line 5, and the path on the server on line 7
- and run using
```
$ make
$ ./fashion_prof.o <OR ./cifar.o>
```

## Layer configuration for FashionMNIST BNN model

L1 Conv &rarr; L2 Maxpool &rarr; L3 Step &rarr; L4 Conv &rarr; L5 Maxpool &rarr; L6 Step &rarr; L7 Flattening &rarr; L8 Gemm &rarr; L9 Step &rarr; L10 Gemm

### Default starting configuration: 
- Layer 1, 2, 4, 5, 8, 10 on GPU using *PROFILE XYZ*
- Layer 3, 6, 7, 9 on CPU.

## Layer configuration for CIFAR-10 BNN model

L1 Conv &rarr; L2 Step &rarr; L3 Conv &rarr; L4 Maxpool &rarr; L5 Step &rarr; L6 Conv &rarr; L7 Step &rarr; L8 Conv &rarr; L9 Maxpool &rarr; L10 Step &rarr; L11 Conv &rarr; L12 Step &rarr; L13 Conv &rarr; L14 Maxpool &rarr; L15 Step &rarr; L16 Flattening &rarr; L17 Gemm &rarr; L18 Step &rarr; L19 Gemm

### Default starting configuration:
- Layer 1, 3, 4, 6, 8, 9, 11, 13, 14, 17, 19 on GPU using *PROFILE XYZ*
- Layer 2, 5, 7, 10, 12, 15, 16, 18 on CPU.

## Run layers on CPU only:
in ```net.cpp``` (```X``` layer number):
-   comment ```/* Layer X GPU */``` lines
-   comment out immediate ```/* Layer X CPU */``` lines

## Run layers on GPU in different parallel configurations:
- Revert previous steps in case they were performed
- Do NOT comment out the STEP layers (i.e. the layers running on CPU by default)

### Available parallel configurations:
- *PROFILE X* – data-images
- *PROFILE Y* – windows
- *PROFILE Z* – neurons
- *PROFILE XY* – data-images + windows
- *PROFILE XZ* – data-images + neurons
- *PROFILE YZ* – windows + neurons
- *PROFILE XYZ* – data-images + windows + neurons

in ```cuda_kernel.cu```:
- Each layer is specifically implemented for every profile iteratively
- Each profile can be found between delimiting lines: ```=============```
- Search for desired profile using CTRL+F, comment out all lines of coude between delimiting lines
- Don't forget to comment-in previously used profiles (e.g. default profile XYZ at the bottom of the file)

