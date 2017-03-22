#!/bin/sh
refresh_log() {
  while true; do
    python $CAFFE_ROOT/tools/extra/parse_log.py log/caffe_train_print.txt log/ 
    sleep 5 
  done
}
refresh_log & 
sleep 1
gnuplot -persist gnuplot_commands
