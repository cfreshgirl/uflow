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

"""UFlow augmentation.
This library contains various augmentation functions.
"""

# pylint:disable=g-importing-member
from mindspore import Tensor, ops, nn
import mindspore
from src.utils import uflow_utils


@ops.constexpr
def construct_tensor(x):
    return Tensor(x)

class BuildSelfupTrans(nn.Cell):
    def __init__(self):
        super(BuildSelfupTrans, self).__init__()
        self.stack = ops.Stack(-1)
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.resize = uflow_utils.ResizeOp()
    def construct(self, images, ij, is_flow,
                  seq_len=2,
                  crop_height=64,
                  crop_width=64,
                  resize=True):
        """Apply augmentations to a list of student images."""

        # Compute random shifts for different images in a sequence.

        i, j = ij

        if is_flow:
            shift_heights = construct_tensor([0, 0])
            shift_widths = construct_tensor([0, 0])
            shifts = self.stack([shift_heights, shift_widths])
            flow_offset = shifts[i] - shifts[j]
            flow_offset = self.reshape(flow_offset, (2, 1, 1))
            images = images + self.cast(flow_offset, mindspore.float32)

        height = images.shape[-2]
        width = images.shape[-1]

        # Assert that the cropped bounding box does not go out of the image frame.
        images = images[:, :, crop_height:height - crop_height, crop_width:width - crop_width]

        if resize:
            images = self.resize(images, height, width, is_flow=is_flow)

        return images
