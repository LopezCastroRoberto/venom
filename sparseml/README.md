Example script to deploy sparse training on resources: ```sparseml.sh```

# 2nd order Pruning Implementations (sparseml/pytorch/sparsification/pruning/)

- ```modifier_pruning_obs.py```: original obs implementation adapted to 8-block pruning

- ```modifier_pruning_obs_nm.py```: one-shot N:M pruning implementation
- ```modifier_pruning_obs_nm_gradual.py```: gradual N:M pruning, m-combinatorial version of canonical vectors
- ```modifier_pruning_nm_pairwise_obs.py```: gradual N:M pruning, pair-wise version of canonical vectors

- ```modifier_pruning_obs_nmv.py```: one-shot V:N:M pruning implementation
- ```modifier_pruning_obs_nmv_gradual.py```: gradual V:N:M pruning, m-combinatorial version of canonical vectors
- ```modifier_nm_pairwise_obs_v.py```: gradual V:N:M pruning, pair-wise version of canonical vectors

# Scripts examples (scripts/)
- ```30epochs_gradual_pruning_squad_block8.sh```: original obs implementation adapted to 8-block pruning

- ```oBERTnm_squad.sh```: one-shot N:M pruning implementation
- ```oBERTnm_squad_gradual.sh```: gradual N:M pruning, m-combinatorial version of canonical vectors
- ```obsnm_squad_gradual_pair.sh```: gradual N:M pruning, m-combinatorial version of canonical vectors

- ```oBERTnmv_squad.sh```: one-shot V:N:M pruning implementation
- ```oBERTnmv_squad_gradual.sh```: gradual V:N:M pruning, m-combinatorial version of canonical vectors
- ```obsnmv_squad_gradual_pair.sh```: gradual V:N:M pruning, pair-wise version of canonical vectors
# Recipes examples (recipes/)
- ```30epochs_8block875_squad.yaml```  original obs implementation adapted to 8-block pruning

- ```oneshot_oBERT_nm.yaml```: one-shot N:M pruning implementation
- ```oBERT_nm_gradual.yaml```: gradual N:M pruning, m-combinatorial version of canonical vectors
- ```obsnm_gradual_pair.yaml```: gradual V:N:M pruning, pair-wise version of canonical vectors

- ```oneshot_oBERT_nmv.yaml```: one-shot V:N:M pruning implementation
- ```oBERT_nmv_gradual.yaml```: gradual V:N:M pruning, m-combinatorial version of canonical vectors
- ```obsnmv_gradual_pair.yaml```: gradual V:N:M pruning, pair-wise version of canonical vectors