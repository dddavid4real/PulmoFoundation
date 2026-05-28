models="PulmoFoundation-E2"

# Loop through each model
for model in $models
do
    # echo "Running evaluations for model: $model"

    studies="External_TCGA_NSCLC"
    ROOT_WSI="path/to/TCGA__NSCLC/pt_files"
    log_dir="logs/${model}"
    
    mkdir -p $log_dir

    for study in $studies
    do
        CUDA_VISIBLE_DEVICES=0 nohup python main.py --model ABMIL \
                                                  --study $study \
                                                  --root ${ROOT_WSI} \
                                                  --feature $model \
                                                  --csv_file dataset_csv/${study}.csv \
                                                  --evaluate \
                                                  --resume './results/results_42/NSCLC/[ABMIL]' \
                                                  --tqdm > "${log_dir}/${study}_${model}.log" 2>&1 &
    done
    
    echo "Launched all jobs for model: $model"

    # studies="External_TCGA_EGFR"
    # ROOT_WSI="path/to/TCGA__NSCLC/pt_files"
    # log_dir="logs/${model}"
    
    # mkdir -p $log_dir
    
    # for study in $studies
    # do
    #     CUDA_VISIBLE_DEVICES=0 nohup python main.py --model ABMIL \
    #                                               --study $study \
    #                                               --root ${ROOT_WSI} \
    #                                               --feature $model \
    #                                               --csv_file dataset_csv/${study}.csv \
    #                                               --evaluate \
    #                                               --resume './results/results_42/EGFR/[ABMIL]' \
    #                                               --tqdm > "${log_dir}/${study}_${model}.log" 2>&1 &
    # done
    
    # echo "Launched all jobs for model: $model"


done

echo "All model evaluations have been launched!"
