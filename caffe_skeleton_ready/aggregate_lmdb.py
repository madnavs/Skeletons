import numpy as np
import sys
import lmdb
import caffe
import sys
import re
import math as math

if len(sys.argv) < 2:
	print("aggregate_lmdb.py filelist_test.txt features_output_folder features_output.csv")
	exit()
else:
	test_fn = sys.argv[1]
	input_lmdb = sys.argv[2]
	output_csv = sys.argv[3]

lines = [line.rstrip('\n').rstrip('\r') for line in open(test_fn)]

lmdb_env = lmdb.open(input_lmdb, readonly=True)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

row = 0

LUT = {}
for key, value in lmdb_cursor:

	if row >= len(lines):
		break

	datum.ParseFromString(value)

	label = datum.label
	data = caffe.io.datum_to_array(datum)

	label1 = data[0][0][0];
	label2 = data[1][0][0];

	full_fn = lines[row]
	split_row = full_fn.split()
	try:
		key = split_row[0]  
		#key = key.split('/')[7].split('.')[0]
		label_gt = split_row[1]
	except:
		print('Invalid filelist format. Update regular expression to aggregate videos')
		print(full_fn)
		exit()

	if key in LUT:    
		LUT[key]['label2'].append(label2)
	else:
		data = {'label_gt': label_gt, 'label2': [label2]}
		LUT[key] = data	
		
	row += 1
	
entroy_eps = np.finfo(float).eps
# Write to output csv - aggregated_key, label_ground_truth, avg_cnn_prediction_2nd_column
with open(output_csv, 'w') as f:
	row = 0
	
	for key, value in LUT.iteritems():
		label2 = value['label2']
		for i in range(len(label2)):

			#all_values = ','.join(str(e) for e in label2)
			str = "%s,%s,%.4f\n" % (key, value['label_gt'], label2[i])
			f.write(str)
		
		row += 1

print '# of Aggregated', row	

