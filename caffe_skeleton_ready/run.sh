#!/bin/sh
#----------------delete prev files ----------------#
rm -r train_image_lmdb
rm -r val_image_lmdb
rm -r test_image_lmdb
rm train_image_mean.binaryproto
rm -r test_features_output
rm trained_models/*
#----------------set caffemodel name ----------------#
MODEL=trained_models/*.caffemodel
#rm $MODEL
#cp pretrained/$MODEL trained_models/
#=trained_models/$MODEL
NUM_LABELS=2

convert_imageset -resize_height 256 -resize_width 256 -shuffle / List/train_list.txt train_image_lmdb
compute_image_mean train_image_lmdb/ train_image_mean.binaryproto

convert_imageset -resize_height 256 -resize_width 256  / List/valid_list.txt val_image_lmdb

convert_imageset -resize_height 256 -resize_width 256  / List/test_list.txt test_image_lmdb

#----------------set solver properties ----------------#
n_test_patches=$(wc -l <  List/test_list.txt)
iter_test=$n_test_patches/64+1
N_train=$(cat List/train_list.txt | wc -l)
batchsize=64
stepsize=$((N_train/batchsize))
echo "# of train patch=$N_train. based on batchsize=$batchsize, step size is $stepsize"
epoch=6
max_iter=$(($stepsize * $epoch))
echo "Updating solver.prototxt with stepsize of $stepsize"
sed -i "s/^stepsize.*$/stepsize: $stepsize/" solver.prototxt
sed -i "s/^max_iter.*$/max_iter: $max_iter/" solver.prototxt
sed -i "s/^snapshot:.*$/snapshot: $max_iter/" solver.prototxt

#-----------------train Start!-----------------------#
screen -dm bash -c 'sleep 5; sh visualize_log.sh;exec sh'
caffe train -solver solver.prototxt -weights ../Step0/$MODEL  2>&1 | tee log/caffe_train_print.txt 

extract_features $MODEL test.prototxt output_3 test_features_output $iter_test lmdb GPU DEVICE_ID=0

python aggregate_lmdb.py List/test_list.txt test_features_output test.csv

if [ $NUM_LABELS = 2 ]; then
               matlab -nodesktop -nosplash -r "Accuracy();quit"
            else
               echo expression evaluated as false
            fi
pkill screen
