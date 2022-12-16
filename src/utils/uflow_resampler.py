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

"""Functions for resampling images."""

#import tensorflow as tf
import mindspore.ops as ops
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import constexpr

@constexpr
def generate_tensor(x, dtype):
    return Tensor(x, dtype)

@constexpr
def generate_tensor_float(x):
    return Tensor(x, mindspore.float32)

@constexpr
def generate_tensor_int(x):
    return Tensor(x, mindspore.int32)

class SafeGatherND(nn.Cell):
    def __init__(self):
        super(SafeGatherND, self).__init__()
        self.transpose = ops.Transpose()
        self.shape = ops.Shape()
        self.zeroslike = ops.ZerosLike()
        self.cast = ops.Cast()
        self.reduceall = ops.ReduceAll()
        self.logicaland = ops.LogicalAnd()
        self.expand_dim = ops.ExpandDims()
        self.gathernd = ops.GatherNd()
    def construct(self, params, indices):
        """Gather slices from params into a Tensor with shape specified by indices.
            Similar functionality to tf.gather_nd with difference: when index is out of
            bound, always return 0.
            Args:
            params: A Tensor. The tensor from which to gather values.
            indices: A Tensor. Must be one of the following types: int32, int64. Index
              tensor.
            Returns:
            A Tensor. Has the same type as params. Values from params gathered from
            specified indices (if they exist) otherwise zeros, with shape
            indices.shape[:-1] + params.shape[indices.shape[-1]:].
            """

        params = self.transpose(params, (0, 2, 3, 1))
        indices = self.transpose(indices, (0, 2, 3, 1))
        params_shape = self.shape(params)
        indices_shape = self.shape(indices)

        slice_dimensions = indices_shape[-1]

        max_index = params_shape[:slice_dimensions] - generate_tensor(1, mindspore.int32)
        min_index = self.zeroslike(max_index)
        min_index = self.cast(min_index, mindspore.int32)

        # max_index = ops.Reshape()(max_index, (indices_shape[-3], 1, 1))
        # min_index = ops.Reshape()(min_index, (indices_shape[-3], 1, 1))
        clipped_indices = ops.clip_by_value(indices, min_index, max_index)
        # Check whether each component of each index is in range [min, max], and
        # allow an index only if all components are in range:
        mask = self.reduceall(
            self.logicaland(indices >= min_index, indices <= max_index), -1)
        mask = self.expand_dim(mask, -1)
        gathernd_tensor = self.gathernd(params, clipped_indices)
        return self.transpose((self.cast(mask, params.dtype) *
                               gathernd_tensor), (0, 3, 1, 2))

class ResampleWithUnstackedWrap(nn.Cell):
    def __init__(self, name='resampler'):
        super(ResampleWithUnstackedWrap, self).__init__()
        self.name = name
        self.shape = ops.Shape()
        self.floor = ops.Floor()
        self.cast = ops.Cast()
        self.sub = ops.Sub()
        self.concat = ops.Concat()
        self.oneslike = ops.OnesLike()
        self.zerolike = ops.ZerosLike()
        self.expand_dims = ops.ExpandDims()
        self.stack = ops.Stack(-3)
        self.gathernd = ops.GatherNd()
        self.safe_gather_nd = SafeGatherND()
    def gather_nd(self, params, indices, safe):
        return (self.safe_gather_nd if safe else self.gathernd)(params, indices)
    def construct(self, data, warp_x, warp_y, safe=True):

        # Compute the four points closest to warp with integer value.
        warp_floor_x = self.floor(warp_x)
        warp_floor_y = self.floor(warp_y)
        # Compute the weight for each point.
        right_warp_weight = warp_x - warp_floor_x
        down_warp_weight = warp_y - warp_floor_y

        warp_floor_x = self.cast(warp_floor_x, mindspore.int32)
        warp_floor_y = self.cast(warp_floor_y, mindspore.int32)
        warp_ceil_x = self.cast(ops.Ceil()(warp_x), mindspore.int32)
        warp_ceil_y = self.cast(ops.Ceil()(warp_y), mindspore.int32)

        left_warp_weight = self.sub(generate_tensor(1.0, right_warp_weight.dtype), right_warp_weight)
        up_warp_weight = self.sub(generate_tensor(1.0, down_warp_weight.dtype), down_warp_weight)
        warp_batch = self.zerolike(warp_y)

        warp_batch = self.cast(warp_batch, mindspore.int32)
        left_warp_weight = self.expand_dims(left_warp_weight, -3)
        down_warp_weight = self.expand_dims(down_warp_weight, -3)
        up_warp_weight = self.expand_dims(up_warp_weight, -3)
        right_warp_weight = self.expand_dims(right_warp_weight, -3)

        up_left_warp = self.stack((warp_batch, warp_floor_y, warp_floor_x))
        up_right_warp = self.stack((warp_batch, warp_floor_y, warp_ceil_x))
        down_left_warp = self.stack((warp_batch, warp_ceil_y, warp_floor_x))
        down_right_warp = self.stack((warp_batch, warp_ceil_y, warp_ceil_x))

        # gather data then take weighted average to get resample result.
        result = (
            (self.gather_nd(data, up_left_warp, safe) * left_warp_weight +
             self.gather_nd(data, up_right_warp, safe) * right_warp_weight) * up_warp_weight +
            (self.gather_nd(data, down_left_warp, safe) * left_warp_weight +
             self.gather_nd(data, down_right_warp, safe) * right_warp_weight) *
            down_warp_weight)

        return result


class Resampler(nn.Cell):
    def __init__(self, name='resampler'):
        super(Resampler, self).__init__()
        self.unstack = ops.Unstack(-3)
        self.name = name
        self.resamplewith = ResampleWithUnstackedWrap()
    def construct(self, data, warp):
        """Resamples input data at user defined coordinates.
           Args:
           data: Tensor of shape `[batch_size, data_height, data_width,
             data_num_channels]` containing 2D data that will be resampled.
           warp: Tensor shape `[batch_size, dim_0, ... , dim_n, 2]` containing the
             coordinates at which resampling will be performed.
           name: Optional name of the op.
           Returns:
           Tensor of resampled values from `data`. The output tensor shape is
           `[batch_size, dim_0, ... , dim_n, data_num_channels]`.
           """

        warp_x, warp_y = self.unstack(warp)
        out = self.resamplewith(data, warp_x, warp_y)
        return out
