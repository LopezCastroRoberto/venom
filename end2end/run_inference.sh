log_file=../result/inference.csv

cd end2end

echo "algo,n,m,v,mean,median,std,len" > $log_file

mv spatha_mod/block_sparse/spmm/blockwise_library.cu spatha_mod/block_sparse/spmm/blockwise_library_v128.cu
mv spatha_mod/block_sparse/spmm/blockwise_library_v64.cu spatha_mod/block_sparse/spmm/blockwise_library.cu
./install.sh
###############
python bert_pytorch.py -m 8 -v 64 >> $log_file

python bert_pytorch.py -m 16 -v 64 >> $log_file

python bert_pytorch.py -m 32 -v 64 >> $log_file
##
##
python gpt2_pytorch.py -m 8 -v 64 >> $log_file

python gpt2_pytorch.py -m 16 -v 64 >> $log_file

python gpt2_pytorch.py -m 32 -v 64 >> $log_file
##
##
python gpt3_pytorch.py -m 8 -v 64 >> $log_file

python gpt3_pytorch.py -m 16 -v 64 >> $log_file

python gpt3_pytorch.py -m 32 -v 64 >> $log_file

###############
mv spatha_mod/block_sparse/spmm/blockwise_library.cu spatha_mod/block_sparse/spmm/blockwise_library_v64.cu
mv spatha_mod/block_sparse/spmm/blockwise_library_v128.cu spatha_mod/block_sparse/spmm/blockwise_library.cu
./install.sh
###############
python bert_pytorch.py -m 8 -v 128 >> $log_file

python bert_pytorch.py -m 16 -v 128 >> $log_file

python bert_pytorch.py -m 32 -v 128 >> $log_file
##
##
python gpt2_pytorch.py -m 8 -v 128 >> $log_file

python gpt2_pytorch.py -m 16 -v 128 >> $log_file

python gpt2_pytorch.py -m 32 -v 128 >> $log_file
##
##
python gpt3_pytorch.py -m 8 -v 128 >> $log_file

python gpt3_pytorch.py -m 16 -v 128 >> $log_file

python gpt3_pytorch.py -m 32 -v 128 >> $log_file
###############