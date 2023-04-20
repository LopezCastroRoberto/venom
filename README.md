<!--[![DOI]()]()-->

# VENOM

<p align="left"><img align="center" width="140" src="venom.png"/></p>


## Build

```
git clone --recurse-submodules git@github.com:LopezCastroRoberto/venom.git
```
```
mkdir build && cd build
```
```
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="86" && make -j 16
```

Note: If you find a problem like this:
```
Policy "CMP0104" is not known to this version of CMake
```
Please, comment this line ```cmake_policy(SET CMP0104 OLD)``` in ```include/sputnik/CMakeLists.txt```

## How to use

Examples:

### Spatha
```
./src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row 8 --m 1024 --k 4096 --n 4096 --d 0.5 --bm 128 --bn 64 --bk 32 --wm 32 --wn 64 --wk 32 --mm 16 --mn 8 --mk 32 --nstage 2 --random --check
```

```
./src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row 16 --m 1024 --k 4096 --n 4096 --d 0.5 --bm 128 --bn 64 --bk 32 --wm 32 --wn 64 --wk 32 --mm 16 --mn 8 --mk 32 --nstage 2 --random --check
```
### cuSparseLt
```
./src/benchmark_spmm --sparsity-type csr --spmm cuSparseLt --gemm cuBlas --precision half --m 1024 --k 4096 --n 768 --d 0.5 --check
```
### CLASP
```
./src/benchmark_spmm --sparsity-type cvs --spmm CLASP --gemm cuBlas --precision half --block-size 16 --m 1024 --k 256 --n 256 --d 0.2 --check
```
## License
Apache-2.0 License

-- Roberto López Castro
--
