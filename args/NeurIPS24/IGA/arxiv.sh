##################################################################################
# 2 0.0005
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains degree\
      --allow_shifts concept\
       --allow_algs IGA\
        --allow_devices 4

conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODArxiv\
   --allow_domains degree\
    --allow_shifts concept\
     --allow_algs IGA\
      --allow_rounds 1 2 3

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODArxiv\
    --allow_domains degree\
     --allow_shifts concept\
      --allow_algs IGA

############################################
# 2 0.0005 / 4 0.0001 / 5 0.005 / 
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains degree\
      --allow_shifts covariate\
       --allow_algs IGA\
        --allow_devices 3

conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODArxiv\
   --allow_domains degree\
    --allow_shifts covariate\
     --allow_algs IGA\
      --allow_rounds 1 2 3

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODArxiv\
    --allow_domains degree\
     --allow_shifts covariate\
      --allow_algs IGA

##################################################################################
# 3 0.0001 / 4 0.01 / 
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains time\
      --allow_shifts concept\
       --allow_algs IGA\
        --allow_devices 6

conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODArxiv\
   --allow_domains time\
    --allow_shifts concept\
     --allow_algs IGA\
      --allow_rounds 1 2 3

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODArxiv\
    --allow_domains time\
     --allow_shifts concept\
      --allow_algs IGA

############################################
# 4 0.0001
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains time\
      --allow_shifts covariate\
       --allow_algs IGA\
        --allow_devices 5

conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODArxiv\
   --allow_domains time\
    --allow_shifts covariate\
     --allow_algs IGA\
      --allow_rounds 1 2 3

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODArxiv\
    --allow_domains time\
     --allow_shifts covariate\
      --allow_algs IGA


conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODArxiv --split degree --shift concept --model IGAGAT

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODArxiv --split degree --shift covariate --model IGAGAT


conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODArxiv --split time --shift concept --model IGAGAT

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODArxiv --split time --shift covariate --model IGAGAT



conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODArxiv --split degree --shift concept --model IGAGCN

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODArxiv --split degree --shift covariate --model GCN


conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODArxiv --split time --shift concept --model GCN

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODArxiv --split time --shift covariate --model GCN