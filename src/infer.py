# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#from caffe2.quantization.server.observer_test import net

"""UFlow: Unsupervised Optical Flow.
This library provides a simple interface for training and inference.
"""
from mindspore import ops, nn, dtype as mstype
from src.utils import uflow_utils

@ops.constexpr
def generate_int(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return int(x.asnumpy())

@ops.constexpr
def generate_str(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return str(x)


class InferNet(nn.Cell):
    """Simple interface with infer and train methods."""

    def __init__(
            self,
            feature_model,
            flow_model,
            checkpoint_dir='',
            num_levels=5,
            occlusion_estimation='wang',
            occ_weights=None,
            occ_thresholds=None,
            occ_clip_max=None
    ):
        """Instantiate a UFlow model.
        Args:
            checkpoint_dir: str, location to checkpoint model
            num_levels: int, how many pwc pyramid layers to use
            occlusion_estimation: which type of occlusion estimation to use
            occ_weights: dict of string -> float indicating how to weight occlusions
            occ_thresholds: dict of str -> float indicating thresholds to apply for
              occlusions
            occ_clip_max: dict of string -> float indicating how to clip occlusion
        Returns:
            Uflow object instance.
        """
        super(InferNet, self).__init__()
        self._feature_model = feature_model
        self._flow_model = flow_model
        self._num_levels = num_levels
        self._occlusion_estimation = occlusion_estimation

        if occ_weights is None:
            occ_weights = {
                'fb_abs': 1.0,
                'forward_collision': 1.0,
                'backward_zero': 10.0
            }
        self._occ_weights = occ_weights

        if occ_thresholds is None:
            occ_thresholds = {
                'fb_abs': 1.5,
                'forward_collision': 0.4,
                'backward_zero': 0.25
            }
        self._occ_thresholds = occ_thresholds

        if occ_clip_max is None:
            occ_clip_max = {'fb_abs': 10.0, 'forward_collision': 5.0}
        self._occ_clip_max = occ_clip_max
        self.stack = ops.Stack()
        self.pow1 = ops.Pow()
        self.ceil = ops.Ceil()
        self.resize_op = uflow_utils.ResizeOp()
        self.reshape = ops.Reshape()
        self.reducesum = ops.ReduceSum(keep_dims=True)
        self.logicaland = ops.LogicalAnd()
        self.cast = ops.Cast()
        self.reducemean = ops.ReduceMean(keep_dims=False)

    def construct(self,
                  image1,
                  image2,
                  flow_gt,
                  occ_mask_gt,
                  input_height=None,
                  input_width=None,
                  resize_flow_to_img_res=True,
                  has_occlusion=True,
                  infer_occlusion=True):

        images = self.stack([image1, image2])[None]
        batch_size, seq_len, image_channels, orig_height, orig_width = images.shape

        if input_height is None:
            input_height = orig_height
        if input_width is None:
            input_width = orig_width

        # Ensure a feasible computation resolution. If specified size is not
        # feasible with the model, change it to a slightly higher resolution.
        # Resize images to desired input height and width.
        if input_height != orig_height or input_width != orig_width:
            images = self.resize_op(
                images, input_height, input_width, is_flow=False)

        images_flattened = self.reshape(
            images,
            (batch_size*seq_len, image_channels, input_height, input_width))
            #(batch_size*seq_len, image_channels, input_height, input_width))

        features_flattened = self._feature_model(
            images_flattened, split_features_by_sample=False)

        features = []
        for f in features_flattened:
            features.append(self.reshape(f, (batch_size, seq_len) + f.shape[1:]))

        features1 = []
        features2 = []

        for f in features:
            features1.append(f[:, 0])
            features2.append(f[:, 1])

        # Compute flow in frame of image1.
        # noinspection PyCallingNonCallable
        flow = self._flow_model(features1, features2, training=False)[0]

        # noinspection PyCallingNonCallable
        # Resize and rescale flow to original resolution. This always needs to be
        # done because flow is generated at a lower resolution.
        if resize_flow_to_img_res:
            flow = self.resize_op(flow, orig_height, orig_width, is_flow=True)


        final_flow = flow
        #final_flow = ops.OnesLike()()
        endpoint_error_occ = self.reducesum((final_flow - flow_gt)**2, -3)**0.5
        #print("endpoint_error_occ", endpoint_error_occ)
        gt_flow_abs = self.reducesum(flow_gt**2, -3)**0.5
        #endpoint_error_occ = tf.reduce_sum(input_tensor=(final_flow - flow_gt)**2, axis=-1, keepdims=True)**0.5
        #gt_flow_abs = tf.reduce_sum(input_tensor=flow_gt**2, axis=-1, keepdims=True)**0.5
        outliers_occ = self.cast(
            self.logicaland(endpoint_error_occ > 3.,
                            endpoint_error_occ > 0.05 * gt_flow_abs), mstype.float32)

        return endpoint_error_occ, outliers_occ
