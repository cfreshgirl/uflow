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
"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np

import mindspore.common.dtype as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.Uflow import PWCFlow, PWCFeaturePyramid

parser = argparse.ArgumentParser(description='What Matters In Unsupervised Optical Flow')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--height', type=int, default=384, help='Image height for training and evaluation.')
parser.add_argument('--width', type=int, default=512, help='Image width for training and evaluation.')
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name1", type=str, default="uflow_feature", help="output file name.")
parser.add_argument("--file_name2", type=str, default="uflow_flow", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")

args = parser.parse_args()


if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    feature_model = PWCFeaturePyramid(
        level1_num_layers=3,
        level1_num_filters=32,
        level1_num_1x1=0,
        original_layer_sizes=False,
        num_levels=5,
        channel_multiplier=1)

    flow_model = PWCFlow(
        dropout_rate=.25,
        normalize_before_cost_volume=True,
        num_levels=5,
        use_feature_warp=True,
        use_cost_volume=True,
        channel_multiplier=1,
        accumulate_flow=True,
        shared_flow_decoder=False)

    assert args.ckpt_file is not None, "config.checkpoint_path is None."
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(feature_model, param_dict)
    load_param_into_net(flow_model, param_dict)

    input_arr1 = Tensor(np.ones([args.batch_size * 2, 3, args.height, args.width]), ms.float32)
    export(feature_model, input_arr1, file_name=args.file_name1, file_format=args.file_format)
    input_arr2 = list([Tensor(np.ones([args.batch_size, 32, args.height//2, args.width//2]), ms.float32),
                       Tensor(np.ones([args.batch_size, 32, args.height//4, args.width//4]), ms.float32),
                       Tensor(np.ones([args.batch_size, 32, args.height//8, args.width//8]), ms.float32),
                       Tensor(np.ones([args.batch_size, 32, args.height//16, args.width//16]), ms.float32),
                       Tensor(np.ones([args.batch_size, 32, args.height//32, args.width//32]), ms.float32)])
    export(flow_model, input_arr2, input_arr2, file_name=args.file_name2, file_format=args.file_format)
