conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains degree\
      --allow_shifts concept\
       --allow_algs ERM\
        --allow_devices 0

# harv
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCora\
    --allow_domains degree\
     --allow_shifts concept\
      --allow_algs ERM


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCora\
   --allow_domains degree\
    --allow_shifts concept\
     --allow_algs ERM\
      --allow_rounds 1 2 3


##################################################################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains degree\
      --allow_shifts concept\
       --allow_algs CIA\
        --allow_devices 0


############################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains degree\
      --allow_shifts covariate\
       --allow_algs CIA\
        --allow_devices 1


##################################################################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains word\
      --allow_shifts concept\
       --allow_algs CIA\
        --allow_devices 2


############################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains word\
      --allow_shifts covariate\
       --allow_algs CIA\
        --allow_devices 3





##################################################################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains degree\
      --allow_shifts concept\
       --allow_algs CIA\
        --allow_devices 8


############################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains degree\
      --allow_shifts covariate\
       --allow_algs CIA\
        --allow_devices 9


##################################################################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains word\
      --allow_shifts concept\
       --allow_algs CIA\
        --allow_devices 6


############################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCora\
     --allow_domains word\
      --allow_shifts covariate\
       --allow_algs CIA\
        --allow_devices 7



conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm CIA --dataset GOODCora --split degree --shift concept --model CIAGAT

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm CIA --dataset GOODCora --split degree --shift covariate --model CIAGAT

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm CIA --dataset GOODCora --split word --shift concept --model CIAGAT

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm CIA --dataset GOODCora --split word --shift covariate --model CIAGAT


conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm CIA --dataset GOODCora --split degree --shift concept --model CIAGCN

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm CIA --dataset GOODCora --split degree --shift covariate --model CIAGCN

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm CIA --dataset GOODCora --split word --shift concept --model CIAGCN

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm CIA --dataset GOODCora --split word --shift covariate --model CIAGCN