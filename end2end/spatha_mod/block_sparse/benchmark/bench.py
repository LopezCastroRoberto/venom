import os.path
import os


dimensions = [
[64   , 64  , 3136],
#[1000 , 2048  ,   1],
[ 512 , 4608  ,  49],
[ 256 , 1024  , 196],
[1024 ,  512  , 196],
[ 256 , 1024  , 196],
[ 256 ,   64  ,3136],
[ 128 ,  512  , 784],
[ 256 , 1024  , 196],
[1024 ,  256  , 196],
[ 256 , 2304  , 196],
[  64 ,  147  ,2544],
[1024 ,  256  , 196],
[ 128 , 1152  , 784],
[ 128 , 1152  , 784],
[ 256 , 2304  , 196],
[ 512 ,  128  , 784],
[ 256 ,   64  ,3136],
[ 128 , 1152  , 784],
[ 256 , 1024  , 196],
[2048 ,  512  ,  49],
[ 128 ,  512  , 784],
[ 256 , 2304  , 196],
[  64 ,  576  ,3136],
[2048 , 1024  ,  49],
[2048 ,  512  ,  49],
[ 256 , 2304  , 196],
[  64 ,  576  ,3136],
[ 512 ,  128  , 784],
[  64 ,  256  ,3136],
[ 256 ,  512  , 784],
[ 128 ,  512  , 784],
[1024 ,  256  , 196],
[1024 ,  256  , 196],
[ 256 , 1024  , 196],
[ 512 , 2048  ,  49],
[ 128 , 1152  , 784],
[ 512 , 4608  ,  49],
[  64 ,  256  ,3136],
[ 512 ,  256  , 784],
[ 128 ,  256  ,3136],
[ 256 , 2304  , 196],
[ 256 ,   64  ,3136],
[ 512 ,  128  , 784],
[ 512 ,  128  , 784],
[  64 ,  576  ,3136],
[ 256 , 2304  , 196],
[ 512 , 1024  , 196],
[ 512 , 4608  ,  49],
[1024 ,  256  , 196],
[1024 ,  256  , 196],
[ 256 ,   64  ,3136],
[ 512 , 2048  ,  49],
[2048 ,  512  ,  49]
]

dim2 = [
[256, 2048, 64],
[512, 4096, 64],
[512, 512, 512],
[1024, 8192, 64],
[1024, 8192, 256]]

def bench(m, k, n, v):
    for gp in [50, 70, 80, 85, 87, 90, 95, 98]:
    #for gp in [50]:
        density = 1-gp/100
        cmd = "/home/roberto.lopez/Documentos/Git/tSPARTAN/build/include/ShflBW_Sparse_NN/block_sparse/benchmark.spmm --random --sparsity-type block --m " +  m + " --k " + k + " --n " + n + " --d " + str(density) + " --block-size " + v

        os.system(cmd)


""" for iter in range(10):
    for v in ["16", "32", "64", "128"]:
        for d in dimensions:
            #bench(str(d[0]), str(d[1]), "512", v)
            bench(str(d[0]), str(d[1]), "256", v)

for iter in range(10):
    for v in ["16", "32", "64", "128"]:
        for d in dim2:
            bench(str(d[0]), str(d[1]), str(d[2]), v) """

for m in range(256,8193,256):
    for n in range(256,8193,256):
        for k in range(256,8192,256):
            for iter in range(10):
                for v in ["16", "32", "64", "128"]:
                    bench(str(m), str(k), str(n), v)