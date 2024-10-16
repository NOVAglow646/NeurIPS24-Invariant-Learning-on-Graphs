##################################################################################
# 2 0.0005
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains degree\
      --allow_shifts concept\
       --allow_algs GTrans\
        --allow_devices 4

############################################
# 2 0.0005 / 4 0.0001 / 5 0.005 / 
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains degree\
      --allow_shifts covariate\
       --allow_algs GTrans\
        --allow_devices 5

##################################################################################
# 3 0.0001 / 4 0.01 / 
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains time\
      --allow_shifts concept\
       --allow_algs GTrans\
        --allow_devices 6
############################################
# 4 0.0001
conda activate /data1/qxwang/conda_envs/pygood
goodtl --sweep_root sweep_configs\
    --launcher MultiLauncher\
    --allow_datasets GOODArxiv\
     --allow_domains time\
      --allow_shifts covariate\
       --allow_algs GTrans\
        --allow_devices 7


conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm GTrans --dataset GOODArxiv --split degree --shift concept --model GTransGAT

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm GTrans --dataset GOODArxiv --split degree --shift covariate --model GTransGAT


conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm GTrans --dataset GOODArxiv --split time --shift concept --model GTransGAT

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm GTrans --dataset GOODArxiv --split time --shift covariate --model GTransGAT



conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm GTrans --dataset GOODArxiv --split degree --shift concept --model GTransGCN

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm GTrans --dataset GOODArxiv --split degree --shift covariate --model GTransGCN


conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm GTrans --dataset GOODArxiv --split time --shift concept --model GTransGCN

conda activate /data1/qxwang/conda_envs/pygood
cd /data1/qxwang/codes/GOOD
python -m GOOD.utils.new_collect_results --algorithm GTrans --dataset GOODArxiv --split time --shift covariate --model GTransGCN