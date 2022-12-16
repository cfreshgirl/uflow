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

"""UFlow utils.
This library contains the various util functions used in UFlow.
"""

import time
import numpy as np
from mindspore import Tensor, ops, nn, dtype as mstype
from mindspore.common.initializer import initializer
from src.utils.uflow_resampler import Resampler

@ops.constexpr
def generate_float(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return float(x)

@ops.constexpr
def generate_tensor_float(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(x, mstype.float32)

@ops.constexpr
def generate_tensor_int(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(x, mstype.int32)

@ops.constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(x)

@ops.constexpr
def construct_list_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(list(range(x)))

class FlowToWrap(nn.Cell):
    def __init__(self):
        super(FlowToWrap, self).__init__()
        self.meshgrid = ops.Meshgrid(indexing="ij")
        self.stack = ops.Stack(0)
        self.linespace = ops.LinSpace()
        self.cast = ops.Cast()
    def construct(self, flow):
        height = flow.shape[-2]
        width = flow.shape[-1]

        i_grid, j_grid = self.meshgrid((
            self.linespace(generate_tensor_float(0.0), generate_tensor_float(height - 1.0), height),
            self.linespace(generate_tensor_float(0.0), generate_tensor_float(width - 1.0), width)))
        grid = self.stack([i_grid, j_grid])
        # Potentially add batch dimension to match the shape of flow.
        if len(flow.shape) == 4:
            grid = grid[None]

        # Add the flow field to the image grid.
        if flow.dtype != grid.dtype:
            grid = self.cast(grid, flow.dtype)
        warp = grid + flow
        return warp

class MaskInvalid(nn.Cell):
    def __init__(self):
        super(MaskInvalid, self).__init__()
        self.logicaland = ops.LogicalAnd()
        self.cast = ops.Cast()
    def construct(self, coords):
        """Mask coordinates outside of the image.
        Valid = 1, invalid = 0.
        Args:
          coords: a 4D float tensor of image coordinates.
        Returns:
          The mask showing which coordinates are valid.
        """
        max_height = generate_float(coords.shape[-2] - 1)
        max_width = generate_float(coords.shape[-1] - 1)
        mask = self.logicaland(
            self.logicaland(coords[:, 0, :, :] >= 0.0,
                            coords[:, 0, :, :] <= max_height),
            self.logicaland(coords[:, 1, :, :] >= 0.0,
                            coords[:, 1, :, :] <= max_width))
        mask = self.cast(mask, mstype.float32)[:, None, :, :]
        return mask


class Resample(nn.Cell):
    def __init__(self):
        super(Resample, self).__init__()
        self.cast = ops.Cast()
        self.stack = ops.Stack()
        self.resampler = Resampler()

    def construct(self, source, coords):
        """Resample the source image at the passed coordinates.
            Args:
              source: tf.tensor, batch of images to be resampled.
              coords: tf.tensor, batch of coordinates in the image.
            Returns:
              The resampled image.
            Coordinates should be between 0 and size-1. Coordinates outside of this range
            are handled by interpolating with a background image filled with zeros in the
            same way that SAME size convolution works.
            """

        # Wrap this function because it uses a different order of height/width dims.
        # if coords is not None:
        orig_source_dtype = source.dtype
        if source.dtype != mstype.float32:
            source = self.cast(source, mstype.float32)
        if coords.dtype != mstype.float32:
            coords = self.cast(coords, mstype.float32)

        coord1 = coords[0][1]
        coord0 = coords[0][0]
        coords_rev = self.stack([coord1, coord0])
        coords_rev = coords_rev[None]
        # print(coords_rev.shape)
        output = self.resampler(source, coords_rev)
        # output = resampler(source, coords[:, ::-1, :, :])
        if orig_source_dtype != source.dtype:
            return self.cast(output, orig_source_dtype)
        return output

class ComputeRangeMap(nn.Cell):
    def __init__(self):
        super(ComputeRangeMap, self).__init__()
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC")
        self.flowtowrap = FlowToWrap()
        self.cast = ops.Cast()
        self.floor = ops.Floor()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.transpose = ops.Transpose()
        self.logicaland = ops.LogicalAnd()
        self.mask_select = ops.MaskedSelect()
        self.concat = ops.Concat()
        self.unsortedsegmentsum = ops.UnsortedSegmentSum()
        self.resizeop = ResizeOp()
        self.flowtowrap = FlowToWrap()
    def construct(self, flow, downsampling_factor=1,
                  reduce_downsampling_bias=True,
                  resize_output=True):
        """Count how often each coordinate is sampled.
        Counts are assigned to the integer coordinates around the sampled coordinates
        using weights from bilinear interpolation.
        Args:
          flow: A float tensor of shape (batch size x height x width x 2) that
            represents a dense flow field.
          downsampling_factor: An integer, by which factor to downsample the output
            resolution relative to the input resolution. Downsampling increases the
            bin size but decreases the resolution of the output. The output is
            normalized such that zero flow input will produce a constant ones output.
          reduce_downsampling_bias: A boolean, whether to reduce the downsampling bias
            near the image boundaries by padding the flow field.
          resize_output: A boolean, whether to resize the output ot the input
            resolution.
        Returns:
          A float tensor of shape [batch_size, height, width, 1] that denotes how
          often each pixel is sampled.
        """

        # Get input shape.
        input_shape = flow.shape
        batch_size, _, input_height, input_width = input_shape

        flow_height = input_height
        flow_width = input_width
        coords = []
        # Apply downsampling (and move the coordinate frame appropriately).
        output_height = input_height // downsampling_factor
        output_width = input_width // downsampling_factor
        if downsampling_factor > 1:
            # Reduce the bias that comes from downsampling, where pixels at the edge
            # will get lower counts that pixels in the middle of the image, by padding
            # the flow field.
            if reduce_downsampling_bias:
                p = downsampling_factor // 2
                flow_height += 2 * p
                flow_width += 2 * p
                # Apply padding in multiple steps to padd with the values on the edge.
                for _ in range(p):
                    flow = self.pad(flow)
                coords = self.flowtowrap(flow) - p
            # Update the coordinate frame to the downsampled one.
            coords = (coords + (1 - downsampling_factor) * 0.5) / downsampling_factor
        elif downsampling_factor == 1:
            coords = self.flowtowrap(flow)
        # Split coordinates into an integer part and a float offset for interpolation.
        coords_floor = self.floor(coords)
        coords_offset = coords - coords_floor
        coords_floor = self.cast(coords_floor, mstype.int32)

        # Define a batch offset for flattened indexes into all pixels.
        batch_range = self.reshape(construct_list_tensor(batch_size), (batch_size, 1, 1))
        idx_batch_offset = self.tile(batch_range, (1, flow_height, flow_width)) * output_height * output_width

        # Flatten everything.
        coords_floor = self.transpose(coords_floor, (0, 2, 3, 1))
        coords_offset = self.transpose(coords_offset, (0, 2, 3, 1))

        coords_floor_flattened = self.reshape(coords_floor, (-1, 2))
        coords_offset_flattened = self.reshape(coords_offset, (-1, 2))
        idx_batch_offset_flattened = self.reshape(idx_batch_offset, (-1,))

        # Initialize results.
        idxs_list = []
        weights_list = []

        # Loop over differences di and dj to the four neighboring pixels.
        for di in range(2):
            for dj in range(2):
                # Compute the neighboring pixel coordinates.
                idxs_i = coords_floor_flattened[:, 0] + di
                idxs_j = coords_floor_flattened[:, 1] + dj

                # Compute the flat index into all pixels.
                idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j

                test = self.logicaland(self.logicaland(idxs_i >= 0, idxs_i < output_height),
                                       self.logicaland(idxs_j >= 0, idxs_j < output_width))

                # valid_idxs = ops.Gather()(idxs, mask, 0)
                # valid_offsets = ops.Gather()(coords_offset_flattened, mask, 0)
                valid_idxs = self.mask_select(idxs, test)
                coords_offset_flattened1 = self.transpose(coords_offset_flattened, (1, 0))[0]
                coords_offset_flattened2 = self.transpose(coords_offset_flattened, (1, 0))[1]
                #valid_offsets0 = self.mask_select(coords_offset_flattened1, test)
                #valid_offsets1 = self.mask_select(coords_offset_flattened2, test)
                # valid_offsets = ops.Stack(1)([valid_offsets0, valid_offsets1])
                # Compute weights according to bilinear interpolation.

                weights_i = (1. - di) - (-1) ** di * coords_offset_flattened1
                weights_j = (1. - dj) - (-1) ** dj * coords_offset_flattened2
                weights_flattened = weights_i * weights_j
                weights = self.mask_select(weights_flattened, test)

                # Append indices and weights to the corresponding list.
                idxs_list.append(valid_idxs)
                weights_list.append(weights)

        # Concatenate everything.
        idxs = self.concat(idxs_list)
        weights = self.concat(weights_list)

        # Sum up weights for each pixel and reshape the result.
        counts = self.unsortedsegmentsum(weights, idxs, batch_size * output_height * output_width)
        count_image = self.reshape(counts, (batch_size, 1, output_height, output_width))

        if downsampling_factor > 1:
            # Normalize the count image so that downsampling does not affect the counts.
            count_image /= downsampling_factor ** 2
            if resize_output:
                count_image = self.resizeop(count_image, input_height, input_width, is_flow=False)

        return count_image

class ComputeWrapsAndOcc(nn.Cell):
    def __init__(self):
        super(ComputeWrapsAndOcc, self).__init__()
        self.flowtowrap = FlowToWrap()
        self.maskinvalid = MaskInvalid()
        self.resample = Resample()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.zerolike = ops.ZerosLike()
        self.cast = ops.Cast()
        self.computerangemap = ComputeRangeMap()
        self.sigmoid = ops.Sigmoid()

    def construct(self, flows, occlusion_estimation,
                  occ_weights=None,
                  occ_thresholds=None,
                  occ_clip_max=None,
                  occlusions_are_zeros=True,
                  occ_active=None):
        """Compute warps, valid warp masks, advection maps, and occlusion masks."""

        warps = []
        range_maps_high_res = []
        range_maps_low_res = []
        occlusion_logits = []
        occlusion_scores = {}
        occlusion_masks = []
        valid_warp_masks = []
        fb_sq_diff = []
        fb_sum_sq = []

        num = len(flows)
        for i in range(num):

            warps.append([])
            range_maps_high_res.append([])
            range_maps_low_res.append([])
            occlusion_masks.append([])
            valid_warp_masks.append([])
            fb_sq_diff.append([])
            fb_sum_sq.append([])

            if len(flows[i]) < 3:
                num1 = len(flows[i])
            else:
                num1 = 3

            for level in range(num1):

                flow_ij = flows[i][level]
                flow_ji = flows[(i + (num // 2)) % num][level]

                # Compute warps (coordinates) and a mask for which coordinates are valid.
                warps[i].append(self.flowtowrap(flow_ij))
                valid_warp_masks[i].append(self.maskinvalid(warps[i][level]))

                # Compare forward and backward flow.
                flow_ji_in_i = self.resample(flow_ji, warps[i][level])
                fb_sq_diff[i].append(
                    self.reduce_sum(
                        (flow_ij + flow_ji_in_i) ** 2, -3))
                fb_sum_sq[i].append(
                    self.reduce_sum(
                        (flow_ij ** 2 + flow_ji_in_i ** 2),
                        -3))
                if level != 0:
                    continue

                # This initializations avoids problems in tensorflow (likely AutoGraph)
                occlusion_mask = self.zerolike(flow_ij[:, :1, ...])
                occlusion_scores['forward_collision'] = self.zerolike(
                    flow_ij[:, :1, ...])
                occlusion_scores['backward_zero'] = self.zerolike(
                    flow_ij[:, :1, ...])
                occlusion_scores['fb_abs'] = self.zerolike(
                    flow_ij[:, :1, ...])

                if occlusion_estimation == 'none' or (
                        occ_active is not None and not occ_active[occlusion_estimation]):
                    occlusion_mask = self.zerolike(flow_ij[:, :1, ...])

                elif occlusion_estimation == 'brox':
                    occlusion_mask = self.cast(
                        fb_sq_diff[i][level] > 0.01 * fb_sum_sq[i][level] + 0.5,
                        mstype.float32)

                elif occlusion_estimation == 'fb_abs':
                    occlusion_mask = self.cast(fb_sq_diff[i][level] ** 0.5 > 1.5, mstype.float32)

                elif occlusion_estimation == 'wang':
                    range_maps_low_res[i].append(
                        self.computerangemap(
                            flow_ji,
                            downsampling_factor=1,
                            reduce_downsampling_bias=False,
                            resize_output=False))

                    # Invert so that low values correspond to probable occlusions,
                    # range [0, 1].
                    occlusion_mask = (
                        1. - ops.clip_by_value(range_maps_low_res[i][level], 0., 1.))
                elif occlusion_estimation == 'wang4':
                    range_maps_low_res[i].append(
                        self.computerangemap(
                            flow_ji,
                            downsampling_factor=4,
                            reduce_downsampling_bias=True,
                            resize_output=True))
                    # Invert so that low values correspond to probable occlusions,
                    # range [0, 1].
                    occlusion_mask = (
                        1. - ops.clip_by_value(range_maps_low_res[i][level], 0., 1.))

                elif occlusion_estimation == 'wangthres':
                    range_maps_low_res[i].append(
                        self.computerangemap(
                            flow_ji,
                            downsampling_factor=1,
                            reduce_downsampling_bias=True,
                            resize_output=True))
                    # Invert so that low values correspond to probable occlusions,
                    # range [0, 1].
                    occlusion_mask = self.cast(range_maps_low_res[i][level] < 0.75,
                                               mstype.float32)

                elif occlusion_estimation == 'wang4thres':
                    range_maps_low_res[i].append(
                        self.computerangemap(
                            flow_ji,
                            downsampling_factor=4,
                            reduce_downsampling_bias=True,
                            resize_output=True))
                    # Invert so that low values correspond to probable occlusions,
                    # range [0, 1].
                    occlusion_mask = self.cast(range_maps_low_res[i][level] < 0.75,
                                               mstype.float32)
                elif occlusion_estimation == 'uflow':
                    # Compute occlusion from the range map of the forward flow, projected
                    # back into the frame of image i. The idea is if many flow vectors point
                    # to the same pixel, those are likely occluded.
                    if 'forward_collision' in occ_weights and (
                            occ_active is None or occ_active['forward_collision']):
                        range_maps_high_res[i].append(
                            self.computerangemap(
                                flow_ij,
                                downsampling_factor=1,
                                reduce_downsampling_bias=True,
                                resize_output=True))
                        fwd_range_map_in_i = self.resample(range_maps_high_res[i][level],
                                                           warps[i][level])
                        # Rescale to [0, max-1].
                        occlusion_scores['forward_collision'] = ops.clip_by_value(
                            fwd_range_map_in_i, 1., occ_clip_max['forward_collision']) - 1.0

                    # Compute occlusion from the range map of the backward flow, which is
                    # already computed in frame i. Pixels that no flow vector points to are
                    # likely occluded.
                    if 'backward_zero' in occ_weights and (occ_active is None or
                                                           occ_active['backward_zero']):
                        range_maps_low_res[i].append(
                            self.computerangemap(
                                flow_ji,
                                downsampling_factor=4,
                                reduce_downsampling_bias=True,
                                resize_output=True))
                        # Invert so that low values correspond to probable occlusions,
                        # range [0, 1].
                        occlusion_scores['backward_zero'] = (
                            1. - ops.clip_by_value(range_maps_low_res[i][level], 0., 1.))

                    # Compute occlusion from forward-backward consistency. If the flow
                    # vectors are inconsistent, this means that they are either wrong or
                    # occluded.
                    if 'fb_abs' in occ_weights and (occ_active is None or
                                                    occ_active['fb_abs']):
                        # Clip to [0, max].
                        occlusion_scores['fb_abs'] = ops.clip_by_value(
                            fb_sq_diff[i][level] ** 0.5, 0.0, occ_clip_max['fb_abs'])

                    occlusion_logits = self.zerolike(flow_ij[:, :1, ...])
                    for k, v in occlusion_scores.items():
                        occlusion_logits += (v - occ_thresholds[k]) * occ_weights[k]
                    occlusion_mask = self.sigmoid(occlusion_logits)

                occlusion_masks[i].append(
                    1. - occlusion_mask if occlusions_are_zeros else occlusion_mask)

        temp = []
        for i in range(num):
            temp.append(range_maps_low_res[(i + num // 2) % num])
        range_maps_low_res = temp

        return warps, valid_warp_masks, range_maps_low_res, occlusion_masks, fb_sq_diff, fb_sum_sq

class ResizeOp(nn.Cell):
    def __init__(self):
        super(ResizeOp, self).__init__()
        self.reshape = ops.Reshape()
        self.mul = ops.Mul()

    def construct(self, img, height, width, is_flow):
        """Resize an image or flow field to a new resolution.
            In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
            performed to account for missing flow entries in the sparse flow field. The
            weighting is based on the resized mask, which determines the 'amount of valid
            flow vectors' that contributed to each individual resized flow vector. Hence,
            multiplying by the reciprocal cancels out the effect of considering non valid
            flow vectors.
            Args:
              img: path to data proto file
              height: int, height of new resolution
              width: int, width of new resolution
              is_flow: bool, flag for scaling flow accordingly
              mask: tf.tensor, mask (optional) per pixel {0,1} flag
            Returns:
              Resized and potentially scaled image or flow field (and mask).
            """
        # Apply resizing at the right shape.
        shape = img.shape
        if len(shape) == 3:
            image = self._resize(img[None], height=height, width=width, is_flow=is_flow)[0]
            #return self._resize(img[None], height=height, width=width, is_flow=is_flow)[0]
        elif len(shape) == 4:
            # Input at the right shape.
            image = self._resize(img, height, width, is_flow)
            #return self._resize(img, height, width, is_flow, mask)
        else:
            # Reshape input to [b, h, w, c], resize and reshape back.
            img_flattened = self.reshape(img, (-1,) + shape[-3:])
            img_resized = self._resize(img_flattened, height, width, is_flow)

            result_img = self.reshape(img_resized, shape[:-3] + img_resized.shape[-3:])
            image = result_img
            #return result_img
        return image

    def _resize(self, img, height, width, is_flow):
        # _, _, orig_height, orig_width = img.shape.as_list()
        orig_height = img.shape[-2]
        orig_width = img.shape[-1]

        if orig_height == height and orig_width == width:
            # early return if no resizing is required
            return img

        # normal resize without anti-alaising
        img_resized = ops.ResizeBilinear((height, width))(img)
        if is_flow:
            # If image is a flow image, scale flow values to be consistent with the
            # new image size.
            scaling = [generate_float(height) / generate_float(orig_height),
                       generate_float(width) / generate_float(orig_width)]
            scaling = self.reshape(construct_tensor(scaling), (1, 2, 1, 1))
            img_resized *= scaling
        return img_resized

class ApplyWrapsStopGrad(nn.Cell):
    def __init__(self):
        super(ApplyWrapsStopGrad, self).__init__()
        self.resample = Resample()
    def construct(self, sources, warps, level):
        """Apply all warps on the correct sources."""

        warped = []
        for i in range(6):
            # Only propagate gradient through the warp, not through the source.
            warped.append(self.resample(
                ops.stop_gradient(sources[(5 - i) // 3]), warps[i][level]))

        return warped

class UpSampele(nn.Cell):
    def __init__(self, is_flow=True):
        super(UpSampele, self).__init__()
        self.cast = ops.Cast()
        self.is_flow = is_flow
    def construct(self, img):
        """Double resolution of an image or flow field.
            Args:
              img: tf.tensor, image or flow field to be resized
              is_flow: bool, flag for scaling flow accordingly
            Returns:
              Resized and potentially scaled image or flow field.
            """
        _, _, height, width = img.shape
        orig_dtype = img.dtype
        if orig_dtype != mstype.float32:
            img = self.cast(img, mstype.float32)
        img_resized = ops.ResizeBilinear((height * 2, width * 2))(img)
        if self.is_flow:
            # Scale flow values to be consistent with the new image size.
            img_resized *= 2
        if img_resized.dtype != orig_dtype:
            return self.cast(img_resized, orig_dtype)
        return img_resized

def robust_l1(x):
    """Robust L1 metric."""
    return (x**2 + 0.001**2)**0.5

def abs_robust_loss(diff, eps=0.01, q=0.4):
    """The so-called robust loss used by DDFlow."""
    return ops.Pow()((ops.Abs()(diff) + eps), q)

def image_grads(image_batch, stride=1):
    image_batch_gh = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
    image_batch_gw = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
    return image_batch_gh, image_batch_gw

def zero_mask_border(mask_bhw3, patch_size):
    """Used to ignore border effects from census_transform."""
    mask_padding = patch_size // 2
    mask = mask_bhw3[:, :, mask_padding:-mask_padding, mask_padding:-mask_padding]
    return ops.Pad(((0, 0), (0, 0), (mask_padding, mask_padding),
                    (mask_padding, mask_padding)))(mask)

class RGBToGrayscale(nn.Cell):
    def __init__(self):
        super(RGBToGrayscale, self).__init__()
        self.transpose = ops.Transpose()
        self.mul = ops.Mul()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.weight = Tensor(np.array([0.2989, 0.5870, 0.1140]), mstype.float16)
        self.w = Tensor(np.array([255]), mstype.float16)
        self.cast = ops.Cast()

    def construct(self, x):
        x = self.transpose(x, (0, 2, 3, 1))
        x = self.cast(x, mstype.float16)
        intensities = self.mul(x, self.weight)
        intensities = self.sum(intensities, -1)
        intensities = self.mul(intensities, self.w)
        intensities = self.transpose(intensities, (0, 3, 1, 2))
        intensities = self.cast(intensities, mstype.float32)
        return intensities

class CensusTransform(nn.Cell):
    def __init__(self, patch_size):
        super(CensusTransform, self).__init__()
        self.rgbtograyscale = RGBToGrayscale()
        self.weight = initializer(ops.Reshape()(
            ops.Eye()(49, 49, mstype.float32),
            (49, 1, 7, 7)), (49, 1, 7, 7))
        self.conv2d = nn.Conv2d(1, patch_size * patch_size, patch_size, pad_mode='same', weight_init=self.weight)
        self.sqrt = ops.Sqrt()
        self.weight = Tensor(np.array([0.2989, 0.5870, 0.1140]), mstype.float32)
        self.w = Tensor(np.array([255]), mstype.float32)
        self.cast = ops.Cast()

    def construct(self, image, patch_size):
        intensities = self.rgbtograyscale(image)
        neighbors = self.conv2d(intensities)
        diff = neighbors - intensities
        # Coefficients adopted from DDFlow.
        diff_norm = diff / self.sqrt(.81 + ops.Square()(diff))
        return diff_norm

class SoftHamming(nn.Cell):
    def __init__(self):
        super(SoftHamming, self).__init__()
        self.square = ops.Square()
        self.reducesum = ops.ReduceSum(keep_dims=True)
        self.w = Tensor(np.array([255]), mstype.float32)
        self.cast = ops.Cast()

    def construct(self, a_bhwk, b_bhwk, thresh=.1):
        sq_dist_bhwk = self.square(a_bhwk - b_bhwk)
        soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
        return self.reducesum(soft_thresh_dist_bhwk, -3)

def soft_hamming(a_bhwk, b_bhwk, thresh=.1):
    """A soft hamming distance between tensor a_bhwk and tensor b_bhwk.
    Args:
        a_bhwk: tf.Tensor of shape (batch, height, width, features)
        b_bhwk: tf.Tensor of shape (batch, height, width, features)
        thresh: float threshold
    Returns:
        a tensor with approx. 1 in (h, w) locations that are significantly
        more different than thresh and approx. 0 if significantly less
        different than thresh.
    """
    sq_dist_bhwk = ops.Square()(a_bhwk - b_bhwk)
    soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
    return ops.ReduceSum(keep_dims=True)(soft_thresh_dist_bhwk, -3)

class CensusLoss(nn.Cell):
    def __init__(self):
        super(CensusLoss, self).__init__()
        self.census_transform = CensusTransform(7)
        self.soft_hamming = SoftHamming()
        self.mul = ops.Mul()
        self.reducesum = ops.ReduceSum()
        self.weight = Tensor(np.array([0.2989, 0.5870, 0.1140]), mstype.float32)
        self.w = Tensor(np.array([255]), mstype.float32)
        self.cast = ops.Cast()

    def construct(self,
                  image_a_bhw3,
                  image_b_bhw3,
                  mask_bhw3,
                  weight,
                  patch_size=7,
                  distance_metric_fn=abs_robust_loss):
        census_image_a_bhwk = self.census_transform(image_a_bhw3, patch_size)
        census_image_b_bhwk = self.census_transform(image_b_bhw3, patch_size)

        hamming_bhw1 = self.soft_hamming(census_image_a_bhwk, census_image_b_bhwk)

        # Set borders of mask to zero to ignore edge effects.
        padded_mask_bhw3 = zero_mask_border(mask_bhw3, patch_size)

        diff = distance_metric_fn(hamming_bhw1)
        diff *= padded_mask_bhw3

        diff_sum = self.reducesum(diff)

        loss_mean = diff_sum / (
            self.reducesum(ops.stop_gradient(padded_mask_bhw3) + 1e-6))

        return loss_mean

class WeightedSsim(nn.Cell):
    def __init__(self):
        super(WeightedSsim, self).__init__()
        self.expanddims = ops.ExpandDims()
        self.avgpool = ops.AvgPool(pad_mode="VALID", kernel_size=3, strides=1)
        self.sum = ops.ReduceSum(keep_dims=True)
        self.weight = Tensor(np.array([0.2989, 0.5870, 0.1140]), mstype.float32)
        self.w = Tensor(np.array([255]), mstype.float32)
        self.cast = ops.Cast()
        self.inf = float('inf')

    def construct(self, x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
        weight = self.expanddims(weight, -1)
        average_pooled_weight = self.avgpool(weight)
        weight_plus_epsilon = weight + weight_epsilon
        inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)
        mu_x = self.avgpool(x * weight_plus_epsilon) * inverse_average_pooled_weight
        mu_y = self.avgpool(y * weight_plus_epsilon) * inverse_average_pooled_weight
        sigma_x = self.avgpool(x ** 2 * weight_plus_epsilon) * inverse_average_pooled_weight - mu_x ** 2
        sigma_y = self.avgpool(y ** 2 * weight_plus_epsilon) * inverse_average_pooled_weight - mu_y ** 2
        sigma_xy = self.avgpool(x * y * weight_plus_epsilon) * inverse_average_pooled_weight - mu_x * mu_y
        if c1 == self.inf:
            ssim_n = (2 * sigma_xy + c2)
            ssim_d = (sigma_x + sigma_y + c2)
        elif c2 == self.inf:
            ssim_n = 2 * mu_x * mu_y + c1
            ssim_d = mu_x ** 2 + mu_y ** 2 + c1
        else:
            ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        result = ssim_n / ssim_d
        return ops.clip_by_value((1 - result) / 2, 0, 1), average_pooled_weight

def time_it(f, num_reps=1, execute_once_before=False):
    """Times a tensorflow function in eager mode.
    Args:
      f: function with no arguments that should be timed.
      num_reps: int, number of repetitions for timing.
      execute_once_before: boolean, whether to execute the function once before
        timing in order to not count the tf.function compile time.
    Returns:
      tuple of the average time in ms and the functions output.
    """
    assert num_reps >= 1
    # Execute f once before timing it to allow tf.function to compile the graph.
    if execute_once_before:
        x = f()
    # Make sure that there is nothing still running on the GPU by waiting for the
    # completion of a bogus command.
    square = ops.Square()
    minval = Tensor(0, dtype=mstype.float32)
    maxval = Tensor(1, dtype=mstype.float32)
    _ = square(ops.uniform((1,), minval, maxval)).asnumpy()
    # Time f for a number of repetitions.
    start_in_s = time.time()
    for _ in range(num_reps):
        x = f()
        # Make sure that f has finished and was not just enqueued by using another
        # bogus command. This will overestimate the computing time of f by waiting
        # until the result has been copied to main memory. Calling reduce_sum
        # reduces that overestimation.
        op = ops.ReduceSum()
        if isinstance(x, (list, tuple)):
            _ = [op(xi).asnumpy() for xi in x]
        else:
            _ = op(x).asnumpy()
    end_in_s = time.time()
    # Compute the average time in ms.
    avg_time = (end_in_s - start_in_s) * 1000. / float(num_reps)
    return avg_time, x
