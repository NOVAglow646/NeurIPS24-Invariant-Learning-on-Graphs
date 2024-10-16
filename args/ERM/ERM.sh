conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODMotif\
     --allow_domains basis\
      --allow_shifts covariate\
       --allow_algs ERM\
        --allow_devices 4 5 7


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODMotif\
     --allow_domains basis\
      --allow_shifts concept\
       --allow_algs ERM\
        --allow_devices 4 5 7

# harv        

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODMotif\
    --allow_domains basis\
     --allow_shifts concept\
      --allow_algs ERM


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODMotif\
   --allow_domains basis\
    --allow_shifts concept\
     --allow_algs ERM\
      --allow_rounds 1 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODMotif\
    --allow_domains basis\
     --allow_shifts covriate\
      --allow_algs GINs


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODMotif\
   --allow_domains basis\
    --allow_shifts covriate\
     --allow_algs GIN\
      --allow_rounds 1 2 3