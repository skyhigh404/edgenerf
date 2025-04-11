# evaluation on 15 scenes:
# scan30,scan34,scan41,scan45, scan82,scan103, scan38, scan21
# scan40, scan55, scan63, scan31, scan8, scan110, scan114,

postfix="_patchsample_bs4096_rbs2048_dsm0.1_nsm0.01_tvnorm0.1"
configs_file="configs/mydtu3.gin"
# for dataset_id in  scan40  scan55  scan63  scan110  scan114  scan21  scan30 scan31 scan8 scan34 scan41 scan45 scan82 scan103 scan38
for dataset_id in scan82 scan103 scan110 scan114
do
    python train.py --gin_configs $configs_file --checkpoint_dir "dtu3_"  --dataset_id $dataset_id --postfix $postfix

    python eval.py --gin_configs $configs_file --checkpoint_dir "dtu3_"  --dataset_id $dataset_id --postfix $postfix
done