#!/usr/bin/env bash
python main.py --gpus 0,1,2,3,4,5,6,7 --method pt_and_ft --pt_method dsm --arch i3d \
--pt_batch_size 256 --pt_workers 32 --pt_spatial_size 224 --pt_stride 4 --pt_data_length 16 \
--pt_nce_k 3569 --pt_softmax \
--pt_moco --pt_epochs 200 --pt_save_freq 4 --pt_print_freq 100 --pt_dataset kinetics \
--pt_train_list ../datasets/lists/kinetics/ssd_kinetics_video_vallist.txt \
--pt_val_list ../datasets/lists/kinetics/ssd_kinetics_video_vallist.txt \
--pt_root /data1/DataSet/kinetics/