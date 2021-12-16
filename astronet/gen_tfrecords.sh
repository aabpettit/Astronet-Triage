#!/bin/bash

set -e

LCDIR=/mnt/tess/lc

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v10-train.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-33-train --num_shards=50

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v10-val.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-33-val --num_shards=5

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v10-test.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-33-test --num_shards=5
