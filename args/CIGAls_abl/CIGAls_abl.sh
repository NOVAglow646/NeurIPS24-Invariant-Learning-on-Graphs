conda activate /data1/qxwang/conda_envs/pygood
为了看：
1. causal contrastive loss反过来（系数-1）会咋样（参数 -1 0 0.3）
2. 只加label smooth spurious loss会咋样（参数 0 0.5 0.3）

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODMotif\
     --allow_domains basis\
      --allow_shifts concept\
       --allow_algs CIGAls_abl\
        --allow_devices 6 7 8

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODMotif\
     --allow_domains basis\
      --allow_shifts covariate\
       --allow_algs CIGAls_abl\
        --allow_devices 4 5 6

# harv cov

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODMotif\
    --allow_domains basis\
     --allow_shifts covariate\
      --allow_algs CIGAls_abl


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODMotif\
   --allow_domains basis\
    --allow_shifts covariate\
     --allow_algs CIGAls_abl\
      --allow_rounds 1 2 3


# harv con

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODMotif\
    --allow_domains basis\
     --allow_shifts concept\
      --allow_algs CIGAls_abl


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODMotif\
   --allow_domains basis\
    --allow_shifts concept\
     --allow_algs CIGAls_abl\
      --allow_rounds 1 2 3