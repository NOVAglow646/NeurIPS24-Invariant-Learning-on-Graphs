conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCBAS\
     --allow_domains color\
      --allow_shifts concept\
       --allow_algs ERM\
        --allow_devices 0

# harv
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCBAS\
    --allow_domains color\
     --allow_shifts concept\
      --allow_algs ERM


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCBAS\
   --allow_domains color\
    --allow_shifts concept\
     --allow_algs ERM\
      --allow_rounds 1 2 3


##################################################################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCBAS\
     --allow_domains color\
      --allow_shifts concept\
       --allow_algs IGA\
        --allow_devices 8

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCBAS\
    --allow_domains color\
     --allow_shifts concept\
      --allow_algs IGA


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCBAS\
   --allow_domains color\
    --allow_shifts concept\
     --allow_algs IGA\
      --allow_rounds 1 2 3

############################################

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODCBAS\
     --allow_domains color\
      --allow_shifts covariate\
       --allow_algs IGA\
        --allow_devices 9

conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
 --final_root final_configs\
  --launcher HarvestLauncher\
   --allow_datasets GOODCBAS\
    --allow_domains color\
     --allow_shifts covariate\
      --allow_algs IGA


conda activate /data1/qxwang/conda_envs/pygood
goodtl --config_root final_configs\
 --launcher HarvestLauncher\
  --allow_datasets GOODCBAS\
   --allow_domains color\
    --allow_shifts covariate\
     --allow_algs IGA\
      --allow_rounds 1 2 3




conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODCBAS --split color --shift concept --model IGAGAT

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODCBAS --split color --shift covariate --model IGAGAT


conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODCBAS --split color --shift concept --model IGAGCN

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm IGA --dataset GOODCBAS --split color --shift covariate --model IGAGCN

