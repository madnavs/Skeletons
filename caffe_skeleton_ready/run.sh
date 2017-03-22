#!/bin/sh
rm -r train_image_lmdb
rm -r val_image_lmdb
rm -r test_image_lmdb
rm train_image_mean.binaryproto
rm val_image_mean.binaryproto
rm test_image_mean.binaryproto
rm trained_models/alexnet.caffemodel
n_test_patches=$(wc -l <  List/test_list.txt)
let iter_test=n_test_patches/256+1
MODEL=alexnet.caffemodel
cp pretrained/$MODEL trained_models/
MODEL=trained_models/$MODEL
NUM_LABELS=2

convert_imageset -resize_height 256 -resize_width 256 -shuffle / List/train_list.txt train_image_lmdb
compute_image_mean train_image_lmdb/ train_image_mean.binaryproto

convert_imageset -resize_height 256 -resize_width 256 -shuffle / List/valid_list.txt val_image_lmdb
compute_image_mean val_image_lmdb/ val_image_mean.binaryproto

convert_imageset -resize_height 256 -resize_width 256 -shuffle / List/test_list.txt test_image_lmdb
compute_image_mean test_image_lmdb/ test_image_mean.binaryproto

screen -d -m ./visualize_log.sh
caffe train -solver solver.prototxt -weights $MODEL  2>&1 | tee log/caffe_train_print.txt 

extract_features $MODEL test.prototxt output_3 test_features_output 10 lmdb GPU DEVICE_ID=0

python aggregate_lmdb.py List/test_list.txt test_features_output test.csv

if [ $NUM_LABELS = 2 ]; then
               matlab -nodesktop -nosplash -r "Accuracy();quit"
            else
               echo expression evaluated as false
            fi

