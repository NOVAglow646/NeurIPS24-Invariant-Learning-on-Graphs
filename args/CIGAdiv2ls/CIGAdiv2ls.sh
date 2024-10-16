conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs --launcher MultiLauncher --allow_datasets GOODCMNIST --allow_domains color --allow_shifts covariate --allow_algs CIGAdiv2ls --allow_devices 8 9

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODMotif\
     --allow_domains basis\
      --allow_shifts concept\
       --allow_algs CIGAdiv2ls\
        --allow_devices 6 7

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODMotif\
     --allow_domains basis\
      --allow_shifts covariate\
       --allow_algs CIGAdiv2ls\
        --allow_devices 7 8

# harv cov

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODMotif\
    --allow_domains basis\
     --allow_shifts covariate\
      --allow_algs CIGAdiv2ls


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODMotif\
   --allow_domains basis\
    --allow_shifts covariate\
     --allow_algs CIGAdiv2ls\
      --allow_rounds 1 2 3


# harv con

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODMotif\
    --allow_domains basis\
     --allow_shifts concept\
      --allow_algs CIGAdiv2ls


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODMotif\
   --allow_domains basis\
    --allow_shifts concept\
     --allow_algs CIGAdiv2ls\
      --allow_rounds 1 2 3