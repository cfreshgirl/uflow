#!/bin/bash
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "========================================================================"
echo "Please run the script as: "
echo "bash run.sh RANK_TABLE"
echo "For example: bash run_distribute_train.sh [TRAIN_DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR] [PRE_TRAINED] [PRE_CKPT_PATH]"
echo "It is better to use the absolute path."
echo "========================================================================"

train_dataset_path=$1
height=$2
width=$3
lr=$4
ckpt_dir=$5

if [ $# == 7 ]; then
  pre_trained=$6
  pre_ckpt_path=$7
fi

ulimit -c unlimited

if [ $# != 5 ] && [ $# != 7 ]; then
  echo "Usage: bash run_distribute_train.sh  [TRAIN_DATASET_PATH] [HEIGHT] [WIDTH] [LR] [CKPT_DIR] [PRE_TRAINED] [PRE_CKPT_PATH]"
  exit 1
fi

export RANK_SIZE=8
echo "rank size = ${RANK_SIZE}"

export GRAPH_OP_RUN=1
export HCCL_WHITELIST_DISABLE=1

cd ../
rm -rf distribute_train
mkdir distribute_train
for f in src *.py
do
    cp -r $f  ./distribute_train
done
cd ./distribute_train || exit
env > env.log

if [ $# == 5 ]; then
  mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
  nohup python -m train --train_on=$train_dataset_path \
    --height=$height \
    --width=$width \
    --lr=$lr \
    --checkpoint_dir=$ckpt_dir \
    --distribute=True > train.log 2>&1 &
else
  mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
    nohup python -m train --train_on=$train_dataset_path \
    --height=$height \
    --width=$width \
    --lr=$lr \
    --checkpoint_dir=$ckpt_dir \
    --distribute=True \
    --pre_trained=$pre_trained \
    --pre_ckpt_path=$pre_ckpt_path > train.log 2>&1 &
fi

cd ../

echo 'uflow training. check it at train.log'
