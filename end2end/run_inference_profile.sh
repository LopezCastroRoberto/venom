
mv spatha_mod/block_sparse/spmm/blockwise_library.cu spatha_mod/block_sparse/spmm/blockwise_library_v128.cu
mv spatha_mod/block_sparse/spmm/blockwise_library_v64.cu spatha_mod/block_sparse/spmm/blockwise_library.cu
./install.sh
###############
python bert_pytorch.py -m 8 -v 64 --profile

python bert_pytorch.py -m 16 -v 64 --profile

python bert_pytorch.py -m 32 -v 64 --profile
##
##
python gpt2_pytorch.py -m 8 -v 64 --profile

python gpt2_pytorch.py -m 16 -v 64 --profile

python gpt2_pytorch.py -m 32 -v 64 --profile
##
##
python gpt3_pytorch.py -m 8 -v 64 --profile

python gpt3_pytorch.py -m 16 -v 64 --profile

python gpt3_pytorch.py -m 32 -v 64 --profile

###############
mv spatha_mod/block_sparse/spmm/blockwise_library.cu spatha_mod/block_sparse/spmm/blockwise_library_v64.cu
mv spatha_mod/block_sparse/spmm/blockwise_library_v128.cu spatha_mod/block_sparse/spmm/blockwise_library.cu
./install.sh
###############
echo "python bert_pytorch.py -m 8 -v 128 --profile"
python bert_pytorch.py -m 8 -v 128 --profile

echo "python bert_pytorch.py -m 16 -v 128 --profile"
python bert_pytorch.py -m 16 -v 128 --profile

echo "python bert_pytorch.py -m 32 -v 128 --profile"
python bert_pytorch.py -m 32 -v 128 --profile
##
##
echo "python gpt2_pytorch.py -m 8 -v 128 --profile"
python gpt2_pytorch.py -m 8 -v 128 --profile

echo "python gpt2_pytorch.py -m 16 -v 128 --profile"
python gpt2_pytorch.py -m 16 -v 128 --profile

echo "python gpt2_pytorch.py -m 32 -v 128 --profile"
python gpt2_pytorch.py -m 32 -v 128 --profile
##
##
echo "python gpt3_pytorch.py -m 8 -v 128 --profile"
python gpt3_pytorch.py -m 8 -v 128 --profile

echo "python gpt3_pytorch.py -m 16 -v 128 --profile"
python gpt3_pytorch.py -m 16 -v 128 --profile

echo "python gpt3_pytorch.py -m 32 -v 128 --profile"
python gpt3_pytorch.py -m 32 -v 128 --profile
###############