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

"""UFlow: Unsupervised Optical Flow.
This library provides a simple interface for training and inference.
"""

import mindspore
from mindspore.ops import stop_gradient
from mindspore import Tensor, ops, nn
from mindspore.common.initializer import initializer
from src import uflow_augmentation
from src.utils import uflow_utils

class UflowLoss(nn.LossBase):
    """
    uflow total loss
    """

    def __init__(self,
                 fb_sigma_teacher,
                 fb_sigma_student,
                 smoothness_edge_weighting,
                 stop_gradient_mask,
                 selfsup_mask,
                 smoothness_at_level,
                 smooth1_weight,
                 census_weight,
                 smoothness_edge_constant):
        """
        Args:
            net: pose net
            cfg: net config
        """
        super(UflowLoss, self).__init__()
        self.fb_sigma_teacher = fb_sigma_teacher
        self.fb_sigma_student = fb_sigma_student
        self.num_pairs = 2.0
        self._smoothness_edge_weighting = smoothness_edge_weighting
        self._stop_gradient_mask = stop_gradient_mask
        self.selfsup_mask = selfsup_mask
        self._smoothness_at_level = smoothness_at_level
        self.zeros = Tensor(0., mindspore.float32)
        self.ones = Tensor(1., mindspore.float32)
        self.smooth1_weight = smooth1_weight
        self.census_weight = census_weight
        self.smoothness_edge_constant = smoothness_edge_constant
        self.weight = initializer(ops.Reshape()(
            ops.Eye()(49, 49, mindspore.float32),
            (49, 1, 7, 7)), (49, 1, 7, 7))
        self.reducesum = ops.ReduceSum()
        self.reducemean = ops.ReduceMean(keep_dims=True)
        self.reducemean1 = ops.ReduceMean()
        self.exp = ops.Exp()
        self.squeeze = ops.Squeeze(-1)
        self.cast = ops.Cast()
        self.oneslike = ops.OnesLike()
        self.equal = ops.Equal()
        self.weighted_ssim = uflow_utils.WeightedSsim()
        self.census_loss = uflow_utils.CensusLoss()
        self.build_selfuptrans = uflow_augmentation.BuildSelfupTrans()
        self.abs = ops.Abs()

    def construct(self, selfsup_weight, images, flows, warps, warped_images, valid_warp_masks,
                  not_occluded_masks, fb_sq_diff, fb_sum_sq):
        """
        Args:
            batch: tf.tensor of shape [b, seq, c, h, w] that holds a batch of image
            sequences.
            weights: dictionary with float entries per loss.
            part_score_targets: part score targets
            part_score_weights: part score weights
            locref_targets: location reference targets
            locref_mask: location reference mask
            pairwise_targets: pairwise targets
            pairwise_mask: pairwise mask
        Return:
            total loss
        """

        # Compute losses.
        loss_total = 0
        losses_photo = 0
        losses_smooth1 = 0
        losses_smooth2 = 0
        losses_ssim = 0
        losses_census = 0
        losses_selfup = 0

        # Count number of non self-sup pairs, for which we will apply the losses.

        # Iterate over image pairs.
        for i in range(6):
            if i not in (1, 4):
                continue

            #if self.ground_truth_occlusions is None:
            if self._stop_gradient_mask:
                mask_level0 = stop_gradient(not_occluded_masks[i][0] *
                                            valid_warp_masks[i][0])
            else:
                mask_level0 = not_occluded_masks[i][0] * valid_warp_masks[i][0]

            edge_constant = self.smoothness_edge_constant

            abs_fn = None
            if self._smoothness_edge_weighting == 'gaussian':
                abs_fn = lambda x: x ** 2
            elif self._smoothness_edge_weighting == 'exponential':
                abs_fn = self.abs

            # Compute image gradients and sum them up to match the receptive field
            # of the flow gradients, which are computed at 1/4 resolution.
            images_level0 = images[i // 3]

            height, width = images_level0.shape[-2:]
            # Resize two times for a smoother result.
            images_level1 = ops.ResizeBilinear((height // 2, width // 2))(images_level0)
            images_level2 = ops.ResizeBilinear((height // 4, width // 4))(images_level1)

            images_at_level = [images_level0, images_level1, images_level2]

            smoothness_at_level = self._smoothness_at_level

            img_gx, img_gy = uflow_utils.image_grads(images_at_level[smoothness_at_level])
            weights_x = self.exp(-self.reducemean(
                (abs_fn(edge_constant * img_gx)),
                -3))
            weights_y = self.exp(-self.reducemean(
                (abs_fn(edge_constant * img_gy)),
                -3))

            # Compute second derivatives of the predicted smoothness.
            flow_gx, flow_gy = uflow_utils.image_grads(flows[i][smoothness_at_level])

            # Compute weighted smoothness
            losses_smooth1 += (
                self.smooth1_weight *
                (self.reducemean1(weights_x * uflow_utils.robust_l1(flow_gx)) +
                 self.reducemean1(weights_y * uflow_utils.robust_l1(flow_gy))) / 2. / self.num_pairs)

            losses_census += self.census_weight * self.census_loss(
                images[i // 3],
                warped_images[i],
                mask_level0,
                weight=self.weight) / self.num_pairs

            _, _, h, w = flows[i][2].shape
            teacher_flow = flows[i - 1][2]
            student_flow = flows[i + 1][2]
            teacher_flow = self.build_selfuptrans(
                images=teacher_flow, ij=(i // 3, 1 - i // 3),
                is_flow=True, crop_height=16, crop_width=16)

            if self.selfsup_mask == 'gaussian':
                student_fb_consistency = self.exp(
                    -fb_sq_diff[i + 1][2] /
                    (self.fb_sigma_student ** 2 * (h ** 2 + w ** 2)))
                teacher_fb_consistency = self.exp(
                    -fb_sq_diff[i - 1][2] / (self.fb_sigma_teacher ** 2 *
                                             (h ** 2 + w ** 2)))

            elif self.selfsup_mask == 'advection':
                student_fb_consistency = not_occluded_masks[i + 1][2]
                teacher_fb_consistency = not_occluded_masks[i - 1][2]
            else:           #if self.selfsup_mask == 'ddflow'
                threshold_student = 0.01 * (fb_sum_sq[
                    i + 1][2]) + 0.5
                threshold_teacher = 0.01 * (fb_sum_sq[
                    i - 1][2]) + 0.5
                student_fb_consistency = self.cast(
                    fb_sq_diff[i + 1][2] < threshold_student,
                    mindspore.float32)
                teacher_fb_consistency = self.cast(
                    fb_sq_diff[i - 1][2] < threshold_teacher,
                    mindspore.float32)

            student_mask = 1. - (
                student_fb_consistency *
                valid_warp_masks[i + 1][2])
            teacher_mask = (
                teacher_fb_consistency *
                valid_warp_masks[i - 1][2])

            teacher_mask = self.build_selfuptrans(
                images=teacher_mask, ij=(i // 3, 1 - i // 3),
                is_flow=False, crop_height=16, crop_width=16)
            error = uflow_utils.robust_l1(stop_gradient(teacher_flow) - student_flow)
            mask = stop_gradient(teacher_mask * student_mask)
            losses_selfup += (
                selfsup_weight * self.reducesum(mask * error) /
                (self.reducesum(self.oneslike(mask)) + 1e-16) / self.num_pairs)

        loss_total += losses_photo
        loss_total += losses_smooth1
        loss_total += losses_smooth2
        loss_total += losses_ssim
        loss_total += losses_census
        loss_total += losses_selfup

        return loss_total

class UflowNetWithLoss(nn.Cell):
    """
    Pack the model network and loss function together to calculate the loss value.
    """
    def __init__(self, feature_model, flow_model, loss):
        super(UflowNetWithLoss, self).__init__()
        self.feature_model = feature_model
        self.flow_model = flow_model
        self.loss = loss
        self._teacher_image_version = 'original'
        self.occlusion_estimation = 'wang'
        self._teacher_feature_model = self.feature_model
        self._teacher_feature_model.set_train(False)
        self.feature_model.set_train()
        self._teacher_flow_model = self.flow_model
        self._teacher_flow_model.set_train(False)
        self.flow_model.set_train()
        self.occ_weights = {'fb_abs': 1000.0,
                            'forward_collision': 1000.0,
                            'backward_zero': 1000.0},
        self.occ_thresholds = {'fb_abs': 1.5,
                               'forward_collision': 0.4,
                               'backward_zero': 0.25},
        self.occ_clip_max = {'fb_abs': 10.0,
                             'forward_collision': 5.0},
        self.occlusions_are_zeros = True
        self.build_selfuptrans = uflow_augmentation.BuildSelfupTrans()
        self.upsample = uflow_utils.UpSampele(is_flow=True)
        self.computewrapsandocc = uflow_utils.ComputeWrapsAndOcc()
        self.applywrapsandstopgrad = uflow_utils.ApplyWrapsStopGrad()

    def construct(self, images_aug, images_without_photo_aug, selfsup_weight, occ_active):

        # conpute flows
        images = []
        flows = []
        flow = []
        features = []
        seq_len = images_aug.shape[1]

        for i in range(seq_len):
            # Populate teacher images with native, unmodified images.
            images.append(images_without_photo_aug[:, i])    #original
            images.append(images_aug[:, i])                #augmented
            images.append(self.build_selfuptrans(     #transformed
                images=images[i * 3 +1], ij=(i, i),
                is_flow=False, crop_height=64, crop_width=64))

        for i, image in enumerate(images):
            # if perform_selfsup and image_version == 'original':
            if i % 3 == 0:
                features.append(self._teacher_feature_model(
                    image, split_features_by_sample=False))
            else:
                features.append(self.feature_model(
                    image, split_features_by_sample=False))

        # Compute flow for all pairs of consecutive images that have the same (or no)
        # transformation applied to them, i.e. that have the same t.
        # pylint:disable=dict-iter-missing-items

        for i, _ in enumerate(features):
            if i % 3 == 1 or i % 3 == 2:
                flow = self.flow_model(
                    features[i], features[(i+3) % 6], training=True)

            else:
                flow = self._teacher_flow_model(
                    features[i], features[(i+3) % 6], training=False)

            # Keep flows at levels 0-2.
            flow_level2 = flow[0]
            flow_level1 = self.upsample(flow_level2)
            flow_level0 = self.upsample(flow_level1)
            flows.append((flow_level0, flow_level1, flow_level2))

        # Prepare images for unsupervised loss (prefer unaugmented images).
        images = []
        seq_len = images_aug.shape[1]
        for i in range(seq_len):
            images.append(images_without_photo_aug[:, i])

        # Warp stuff and compute occlusion.

        warps, valid_warp_masks, _, not_occluded_masks, fb_sq_diff, fb_sum_sq = self.computewrapsandocc(
            flows,
            occlusion_estimation=self.occlusion_estimation,
            occ_weights=self.occ_weights,
            occ_thresholds=self.occ_thresholds,
            occ_clip_max=self.occ_clip_max,
            occlusions_are_zeros=True,
            occ_active=occ_active)
        # Warp images and features.
        warped_images = self.applywrapsandstopgrad(images, warps, level=0)

        return self.loss(selfsup_weight, images, flows, warps, warped_images, valid_warp_masks,
                         not_occluded_masks, fb_sq_diff, fb_sum_sq)
