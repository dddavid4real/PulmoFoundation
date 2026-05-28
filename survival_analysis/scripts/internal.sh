features='PulmoFoundation-E2'
lr=2e-4

for feature in $features
do

    echo "Running internal validation for model: $feature"

    studies="LUAD LUSC"  # LUSC
    feature_path="path/to/TCGA__NSCLC/pt_files"
    log_dir="logs/${feature}"

    mkdir -p "$log_dir"

    for study in $studies
    do
        echo "study: $study | model: $feature"

        CUDA_VISIBLE_DEVICES=2 python main.py \
            --model AttMIL \
            --csv_file ./dataset_csv/${study}.csv \
            --feature_path $feature_path \
            --feature $feature \
            --study $study \
            --modal WSI \
            --num_epoch 20 \
            --batch_size 1 \
            --lr $lr > "${log_dir}/${study}_${feature}.log" 2>&1 &
    done

done

echo "All training jobs have been completed!"
