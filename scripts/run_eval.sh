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
echo "For example: bash run_eval.sh [EVAL_DATASET_PATH] [HEIGHT] [WIDTH] [CKPT_PATH] [DEVICE_ID]"
echo "It is better to use the absolute path."
echo "========================================================================"

eval_dataset_path=$1
height=$2
width=$3
ckpt_path=$4
device_id=$5
ulimit -c unlimited

if [ $# != 5 ]; then
  echo "Usage: bash run_eval.sh [EVAL_DATASET_PATH] [HEIGHT] [WIDTH] [CKPT_PATH] [DEVICE_ID]"
  exit 1
fi

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export device_id=$device_id
export RANK_ID=$device_id
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd ..

echo "eval begin"
nohup python -m eval --eval_on=$eval_dataset_path \
  --height=$height \
  --width=$width \
  --checkpoint_dir=$ckpt_path \
  --device_id=$device_id > eval.log 2>&1 &

echo 'uflow eval. check it at eval.log'
