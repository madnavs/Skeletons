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

convert_imageset  -shuffle / List/train_list.txt train_image_lmdb
compute_image_mean train_image_lmdb/ train_image_mean.binaryproto

convert_imageset   / List/valid_list.txt val_image_lmdb

convert_imageset   / List/test_list.txt test_image_lmdb

#----------------set solver properties ----------------#
batchsize=64 ##Chage this!!
epoch=10
NUM_LABELS=2
##-------------val list --------------------
n_val_patches=$(wc -l <  List/valid_list.txt)
iter_val=$((n_val_patches/batchsize+1))
sed -i "s/^test_iter.*$/test_iter: $iter_val/" solver.prototxt

##--------------Test List Prop -------------
n_test_patches=$(wc -l <  List/test_list.txt)
iter_test=$((n_test_patches/batchsize+1))
N_train=$(cat List/train_list.txt | wc -l)
stepsize=$((N_train/batchsize))
echo "# of train patch=$N_train. based on batchsize=$batchsize, step size is $stepsize"

max_iter=$(($stepsize * $epoch))
echo "Updating solver.prototxt with stepsize of $stepsize"
sed -i "s/^stepsize.*$/stepsize: $stepsize/" solver.prototxt
sed -i "s/^max_iter.*$/max_iter: $max_iter/" solver.prototxt
sed -i "s/^snapshot:.*$/snapshot: $max_iter/" solver.prototxt

#-----------------train Start!-----------------------#
screen -dm bash -c 'sleep 5; sh visualize_log.sh;exec sh'
caffe train -solver solver.prototxt -weights ../Step0/$MODEL  2>&1 | tee log/caffe_train_print.txt 

echo "using this model" $MODEL
extract_features $MODEL test.prototxt output_3 test_features_output $iter_test lmdb GPU DEVICE_ID=0

python aggregate_lmdb.py List/test_list.txt test_features_output test.csv

if [ $NUM_LABELS = 2 ]; then
               matlab -nodesktop -nosplash -r "plot_aggregate();quit"
            else
               echo expression evaluated as false
            fi
pkill screen
