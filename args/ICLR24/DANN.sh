conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
tensorboard --logdir runs

############ Cora degree
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains degree\
      --allow_shifts covariate\
       --allow_algs DANN\
        --allow_devices 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains degree\
      --allow_shifts concept\
       --allow_algs DANN\
        --allow_devices 2 5

# harv        

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCora\
    --allow_domains degree\
     --allow_shifts concept\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCora\
   --allow_domains degree\
    --allow_shifts concept\
     --allow_algs DANN\
      --allow_rounds 1 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCora\
    --allow_domains degree\
     --allow_shifts covariate\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCora\
   --allow_domains degree\
    --allow_shifts covariate\
     --allow_algs DANN\
      --allow_rounds 1 2 3





############ Cora word
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains word\
      --allow_shifts covariate\
       --allow_algs DANN\
        --allow_devices 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains word\
      --allow_shifts concept\
       --allow_algs DANN\
        --allow_devices 2 5
# harv        

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCora\
    --allow_domains word\
     --allow_shifts concept\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCora\
   --allow_domains word\
    --allow_shifts concept\
     --allow_algs DANN\
      --allow_rounds 1 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCora\
    --allow_domains word\
     --allow_shifts covariate\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCora\
   --allow_domains word\
    --allow_shifts covariate\
     --allow_algs DANN\
      --allow_rounds 1 2 3



############ Arxiv degree
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains degree\
      --allow_shifts covariate\
       --allow_algs DANN\
        --allow_devices 1


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains degree\
      --allow_shifts concept\
       --allow_algs DANN\
        --allow_devices 1

# harv        

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODArxiv\
    --allow_domains degree\
     --allow_shifts concept\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODArxiv\
   --allow_domains degree\
    --allow_shifts concept\
     --allow_algs DANN\
      --allow_rounds 1 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODArxiv\
    --allow_domains degree\
     --allow_shifts covariate\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODArxiv\
   --allow_domains degree\
    --allow_shifts covariate\
     --allow_algs DANN\
      --allow_rounds 1 2 3





############ Arxiv time
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains time\
      --allow_shifts covariate\
       --allow_algs DANN\
        --allow_devices 5


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains time\
      --allow_shifts concept\
       --allow_algs DANN\
        --allow_devices 1 3

# harv        

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODArxiv\
    --allow_domains time\
     --allow_shifts concept\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODArxiv\
   --allow_domains time\
    --allow_shifts concept\
     --allow_algs DANN\
      --allow_rounds 1 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODArxiv\
    --allow_domains time\
     --allow_shifts covariate\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODArxiv\
   --allow_domains time\
    --allow_shifts covariate\
     --allow_algs DANN\
      --allow_rounds 1 2 3



#################################################################GOODCBAS color


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCBAS\
     --allow_domains color\
      --allow_shifts covariate\
       --allow_algs DANN\
        --allow_devices 4

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCBAS\
     --allow_domains color\
      --allow_shifts concept\
       --allow_algs DANN\
        --allow_devices 1
############################# harv

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCBAS\
    --allow_domains color\
     --allow_shifts covariate\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCBAS\
   --allow_domains color\
    --allow_shifts covariate\
     --allow_algs DANN\
      --allow_rounds 1 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCBAS\
    --allow_domains color\
     --allow_shifts concept\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCBAS\
   --allow_domains color\
    --allow_shifts concept\
     --allow_algs DANN\
      --allow_rounds 1 2 3





#################################################################GOODTwitch language


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODTwitch\
     --allow_domains language\
      --allow_shifts covariate\
       --allow_algs DANN\
        --allow_devices 7

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODTwitch\
     --allow_domains language\
      --allow_shifts concept\
       --allow_algs DANN\
        --allow_devices 5
############################# harv

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODTwitch\
    --allow_domains language\
     --allow_shifts covariate\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODTwitch\
   --allow_domains language\
    --allow_shifts covariate\
     --allow_algs DANN\
      --allow_rounds 1 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODTwitch\
    --allow_domains language\
     --allow_shifts concept\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODTwitch\
   --allow_domains language\
    --allow_shifts concept\
     --allow_algs DANN\
      --allow_rounds 1 2 3




#################################################################GOODWebKB university


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODWebKB\
     --allow_domains university\
      --allow_shifts covariate\
       --allow_algs DANN\
        --allow_devices 2

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODWebKB\
     --allow_domains university\
      --allow_shifts concept\
       --allow_algs DANN\
        --allow_devices 2
############################# harv

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODWebKB\
    --allow_domains university\
     --allow_shifts covariate\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODWebKB\
   --allow_domains university\
    --allow_shifts covariate\
     --allow_algs DANN\
      --allow_rounds 1 2 3


conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODWebKB\
    --allow_domains university\
     --allow_shifts concept\
      --allow_algs DANN


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODWebKB\
   --allow_domains university\
    --allow_shifts concept\
     --allow_algs DANN\
      --allow_rounds 1 2 3