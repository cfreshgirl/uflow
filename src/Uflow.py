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

# Lint as: python3

"""UFlow models.
This library contains the models used in UFlow. Our model is a slightly modified
version of the PWC net by Sun et al (https://arxiv.org/pdf/1709.02371.pdf).
In particular, we change the number of layers and filters in the feature
pyramid, we introduce a pyramid-level dropout, and we normalize the features
before computing a cost volume. We found these changes to improve the
performance.
"""

# Lint as: python3

import mindspore
from mindspore import Tensor, ops, nn
from mindspore.ops import constexpr
from src.utils import uflow_utils

@constexpr
def generate_tensor_int(x):
    return Tensor(x, mindspore.int32)

@constexpr
def generate_tensor_float(x):
    return Tensor(x, mindspore.float32)

@constexpr
def uniform():
    return ops.uniform((1,), Tensor(0.), Tensor(1.))[0]

class NormalizeFeatures(nn.Cell):
    def __init__(self):
        super(NormalizeFeatures, self).__init__()
        self.reduce_mean = ops.ReduceMean()
        self.stack = ops.Stack()
        self.sqrt = ops.Sqrt()

    def construct(self, feature_list, normalize, center, moments_across_channels,
                  moments_across_images):
        """Normalizes feature tensors (e.g., before computing the cost volume).
            Args:
              feature_list: list of tf.tensors, each with dimensions [b, h, w, c]
              normalize: bool flag, divide features by their standard deviation
              center: bool flag, subtract feature mean
              moments_across_channels: bool flag, compute mean and std across channels
              moments_across_images: bool flag, compute mean and std across images
            Returns:
              list, normalized feature_list
            """

        # Compute feature statistics.
        means = []
        variances = []

        # statistics = collections.defaultdict(list)
        axes = (-2, -1, -3) if moments_across_channels else (-2, -1)
        for feature_image in feature_list:
            mean, variance = nn.Moments(axis=axes, keep_dims=True)(feature_image)
            means.append(mean)
            variances.append(variance)

        if moments_across_images:
            means = ([self.reduce_mean(self.stack(means))] * len(feature_list))
            variances = ([self.reduce_mean(self.stack(variances))] * len(feature_list))

        stds = []
        for v in variances:
            stds.append(self.sqrt(v + 1e-16))

        # Center and normalize features.
        temp = []
        if center:
            for f, mean in zip(feature_list, means):
                temp.append(f - mean)
            feature_list = temp
        temp = []
        if normalize:
            for f, std in zip(feature_list, stds):
                temp.append(f / std)
            feature_list = temp
        return feature_list

class ComputeCostVolume(nn.Cell):
    def __init__(self):
        super(ComputeCostVolume, self).__init__()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.concat = ops.Concat(-3)
    def construct(self, features1, features2, max_displacement):
        """Compute the cost volume between features1 and features2.
            Displace features2 up to max_displacement in any direction and compute the
            per pixel cost of features1 and the displaced features2.
            Args:
              features1: tf.tensor of shape [b, h, w, c]
              features2: tf.tensor of shape [b, h, w, c]
              max_displacement: int, maximum displacement for cost volume computation.
            Returns:
              tf.tensor of shape [b, h, w, (2 * max_displacement + 1) ** 2] of costs for
              all displacements.
            """

        # Set maximum displacement and compute the number of image shifts.
        _, _, height, width = features1.shape
        # print(max_displacement)
        # if max_displacement <= 0 or max_displacement >= height:
        # raise ValueError(f'Max displacement of {max_displacement} is too large.')

        max_disp = max_displacement
        num_shifts = 2 * max_disp + 1

        # Pad features2 and shift it while keeping features1 fixed to compute the
        # cost volume through correlation.

        # Pad features2 such that shifts do not go out of bounds.
        features2_padded = ops.Pad(((0, 0), (0, 0), (max_disp, max_disp), (max_disp, max_disp)))(features2)
        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):
                corr = self.reduce_mean(
                    features1 * features2_padded[:, :, i:(height + i), j:(width + j)], -3)
                cost_list.append(corr)
        cost_volume = self.concat(cost_list)
        return cost_volume

class PWCFlow(nn.Cell):
    """Model for estimating flow based on the feature pyramids of two images."""

    def __init__(self,
                 leaky_relu_alpha=0.1,
                 dropout_rate=0.25,
                 num_channels_upsampled_context=32,
                 num_levels=5,
                 normalize_before_cost_volume=True,
                 channel_multiplier=1.,
                 use_cost_volume=True,
                 use_feature_warp=True,
                 accumulate_flow=True,
                 shared_flow_decoder=False,
                 action_channels=None):

        super(PWCFlow, self).__init__()
        self._leaky_relu_alpha = leaky_relu_alpha
        self._drop_out_rate = dropout_rate
        self._num_context_up_channels = num_channels_upsampled_context
        self._num_levels = num_levels
        self._normalize_before_cost_volume = normalize_before_cost_volume
        self._channel_multiplier = channel_multiplier
        self._use_cost_volume = use_cost_volume
        self._use_feature_warp = use_feature_warp
        self._accumulate_flow = accumulate_flow
        self._shared_flow_decoder = shared_flow_decoder
        self._action_channels = action_channels

        self._refine_model = self._build_refinement_model()

        self.leakyrelu = nn.LeakyReLU(alpha=self._leaky_relu_alpha)
        self._flow_layers = self._build_flow_layers()

        self.leakyrelu = nn.LeakyReLU(alpha=self._leaky_relu_alpha)
        if not self._use_cost_volume:
            self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs()
        if num_channels_upsampled_context:
            self._context_up_layers = self._build_upsample_layers(
                num_channels=int(num_channels_upsampled_context * channel_multiplier))
        self.zeros = ops.Zeros()
        self.concat = ops.Concat(-3)
        self.cast = ops.Cast()
        self.greater = ops.Greater()
        self.concat_1 = ops.Concat(1)
        self.normalize_feature = NormalizeFeatures()
        self.computcostvolume = ComputeCostVolume()
        self.flowtowrap = uflow_utils.FlowToWrap()
        self.resample = uflow_utils.Resample()
        self.upsample = uflow_utils.UpSampele()
        self.one = Tensor(1., mindspore.float32)
        self.zero = Tensor(0., mindspore.float32)

    def construct(self, feature_pyramid1, feature_pyramid2, training=False):
        """Run the model."""
        context = None
        flow = None
        flow_up = None
        context_up = None
        flows = []
        warped2 = []
        reverse_list = []
        features1 = feature_pyramid1
        features2 = feature_pyramid2

        if isinstance(features1, Tensor):
            feature_pyramid1 = []
            feature_pyramid2 = []
            num, batch_size, channels, height, width = features1.shape
            for i in range(num):
                feature = features1[i].resize(batch_size, channels, height//(2 ** i), width//(2 ** i))
                feature_pyramid1.append(feature)
                feature = features2[i].resize(batch_size, channels, height//(2 ** i), width//(2 ** i))
                feature_pyramid2.append(feature)

        now_list = enumerate(zip(feature_pyramid1, feature_pyramid2))[1:]
        #now_list = list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]

        for i in range(len(now_list)):
            reverse_list.append(now_list[len(now_list) - i - 1])

        # Go top down through the levels to the second to last one to estimate flow.
        for level, (features1, features2) in reverse_list:

            batch_size, _, height, width = features1.shape
            # init flows with zeros for coarsest level if needed

            # Warp features2 with upsampled flow from higher level.
            if flow_up is None or not self._use_feature_warp:
                warped2 = features2
            else:
                warp_up = self.flowtowrap(flow_up)
                warped2 = self.resample(features2, warp_up)
            #print("warped2=======", warped2)
            # Compute cost volume by comparing features1 and warped features2.
            features1_normalized, warped2_normalized = self.normalize_feature(
                [features1, warped2],
                normalize=self._normalize_before_cost_volume,
                center=self._normalize_before_cost_volume,
                moments_across_channels=True,
                moments_across_images=True)

            cost_volume = self.computcostvolume(
                features1_normalized, warped2_normalized, max_displacement=4)

            # Compute context and flow from previous flow, cost volume, and features1.
            if flow_up is None:
                x_in = self.concat((cost_volume, features1))
            else:
                if context_up is None:
                    x_in = self.concat((flow_up, cost_volume, features1))
                else:
                    x_in = self.concat(
                        (context_up, flow_up, cost_volume, features1))

            x_out = None

            for i in range(5):
                x_out = self._flow_layers[(level-1)*6+i](x_in)
                x_in = self.concat([x_in, x_out])
            context = x_out
            flow = self._flow_layers[(level-1)*6+5](context)

            if (training and self._drop_out_rate):
                maybe_dropout = self.cast(

                    self.greater(ops.uniform((1,), self.zero, self.one)[0], self._drop_out_rate),
                    mindspore.float32)
                context *= maybe_dropout
                flow *= maybe_dropout

            if flow_up is not None and self._accumulate_flow:
                flow += flow_up

            # Upsample flow for the next lower level.
            flow_up = self.upsample(flow)

            context_up = self._context_up_layers[level-1](context)

            # Append results to list.
            flows.insert(0, flow)

        # Refine flow at level 1.
        refinement = self._refine_model(self.concat_1((context, flow)))
        if (training and self._drop_out_rate):
            refinement *= self.cast(
                #self.greater(ops.uniform((), generate_tensor_float(0), generate_tensor_float(1)), self._drop_out_rate),
                self.greater(ops.uniform((1,), self.zero, self.one)[0], self._drop_out_rate),
                mindspore.float32)
        refined_flow = flow + refinement
        flows[0] = refined_flow

        return flows

    def _build_cost_volume_surrogate_convs(self):
        layers = []
        for _ in range(self._num_levels):
            layers.append(
                nn.Conv2d(
                    in_channels=int(64 * self._channel_multiplier),
                    out_channels=int(64 * self._channel_multiplier),
                    kernel_size=(4, 4),
                    pad_mode='same',
                    data_format="NCHW",
                    has_bias=True,
                    weight_init="XavierUniform"))
        return layers

    def _build_upsample_layers(self, num_channels):
        """Build layers for upsampling via deconvolution."""
        layers = nn.CellList()
        for unused_level in range(self._num_levels - 1):
            layers.append(
                nn.Conv2dTranspose(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(4, 4),
                    stride=2,
                    pad_mode='same',
                    has_bias=True,
                    weight_init="XavierUniform"))
        return layers


    def _build_flow_layers(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        block_layers = [128, 128, 96, 64, 32]
        layers = nn.CellList()
        for i in range(1, self._num_levels):
            last_in_channels = (64 + 32) if not self._use_cost_volume else (81 + 32)
            # last_in_channels = 32
            if self._action_channels is not None and self._action_channels > 0:
                last_in_channels += self._action_channels + 2  # 2 for xy augmentation
            if i != self._num_levels - 1:
                last_in_channels += 2 + self._num_context_up_channels

            for c in block_layers:
                layers.append(
                    nn.SequentialCell([
                        nn.Conv2d(
                            in_channels=last_in_channels,
                            out_channels=int(c * self._channel_multiplier),
                            kernel_size=(3, 3),
                            stride=1,
                            pad_mode='same',
                            data_format="NCHW",
                            has_bias=True,
                            weight_init="XavierUniform"),
                        nn.LeakyReLU(
                            alpha=self._leaky_relu_alpha)
                    ]))
                last_in_channels += int(c * self._channel_multiplier)
            layers.append(
                nn.Conv2d(
                    in_channels=block_layers[-1],
                    out_channels=2,
                    kernel_size=(3, 3),
                    stride=1,
                    pad_mode='same',
                    data_format="NCHW",
                    has_bias=True,
                    weight_init="XavierUniform"))

        return layers

    def _build_refinement_model(self):
        """Build model for flow refinement using dilated convolutions."""
        layers = []
        last_in_channels = 32 + 2
        # layers.append(ops.Concat(-1))
        for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
            layers.append(
                nn.Conv2d(
                    in_channels=last_in_channels,
                    out_channels=c * self._channel_multiplier,
                    kernel_size=(3, 3),
                    stride=1,
                    pad_mode='same',
                    dilation=d,
                    has_bias=True,
                    weight_init="XavierUniform"))
            layers.append(
                nn.LeakyReLU(alpha=self._leaky_relu_alpha))
            last_in_channels = c * self._channel_multiplier
        layers.append(
            nn.Conv2d(
                in_channels=last_in_channels,
                out_channels=2,
                kernel_size=(3, 3),
                stride=1,
                pad_mode='same',
                has_bias=True,
                weight_init="XavierUniform"))
        return nn.SequentialCell(layers)


    def _build_1x1_shared_decoder(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.CellList()
        for _ in range(1, self._num_levels):
            result.append(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(1, 1),
                    stride=1,
                    pad_mode='same',
                    data_format="NCHW",
                    has_bias=True,
                    weight_init="XavierUniform"))
        return result


class PWCFeaturePyramid(nn.Cell):
    """Model for computing a feature pyramid from an image."""

    def __init__(self,
                 leaky_relu_alpha=0.1,
                 filters=None,
                 level1_num_layers=3,
                 level1_num_filters=16,
                 level1_num_1x1=0,
                 original_layer_sizes=False,
                 num_levels=5,
                 channel_multiplier=1.,
                 num_channels=3):
        """Constructor.
        Args:
          leaky_relu_alpha: Float. Alpha for leaky ReLU.
          filters: Tuple of tuples. Used to construct feature pyramid. Each tuple is
            of form (num_convs_per_group, num_filters_per_conv).
          level1_num_layers: How many layers and filters to use on the first
            pyramid. Only relevant if filters is None and original_layer_sizes
            is False.
          level1_num_filters: int, how many filters to include on pyramid layer 1.
            Only relevant if filters is None and original_layer_sizes if False.
          level1_num_1x1: How many 1x1 convolutions to use on the first pyramid
            level.
          original_layer_sizes: bool, if True, use the original PWC net number
            of layers and filters.
          num_levels: int, How many feature pyramid levels to construct.
          channel_multiplier: float, used to scale up or down the amount of
            computation by increasing or decreasing the number of channels
            by this factor.
          pyramid_resolution: str, specifies the resolution of the lowest (closest
            to input pyramid resolution)
          use_bfloat16: bool, whether or not to run in bfloat16 mode.
        """

        super(PWCFeaturePyramid, self).__init__()
        self._channel_multiplier = channel_multiplier
        if num_levels > 6:
            raise NotImplementedError('Max number of pyramid levels is 6')
        if filters is None:
            if original_layer_sizes:
                # Orig - last layer
                filters = ((3, 16), (3, 32), (3, 64), (3, 96), (3, 128),
                           (3, 196))[:num_levels]
            else:
                filters = ((level1_num_layers, level1_num_filters), (3, 32), (3, 32),
                           (3, 32), (3, 32), (3, 32))[:num_levels]
        assert filters
        assert all(len(t) == 2 for t in filters)
        assert all(t[0] > 0 for t in filters)

        self._leaky_relu_alpha = leaky_relu_alpha
        self._convs = []
        self._level1_num_1x1 = level1_num_1x1

        self._conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv6 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv7 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv8 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv9 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv10 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv11 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv12 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv13 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv14 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")
        self._conv15 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            pad_mode='valid',
            data_format="NCHW",
            has_bias=True,
            weight_init="XavierUniform")

        self.pad = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha=self._leaky_relu_alpha)

    def construct(self, x, split_features_by_sample=False):
        x = x * 2. - 1.  # Rescale input from [0,1] to [-1, 1]
        features = []

        x = self.pad(x)
        x = self._conv1(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv2(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv3(x)
        x = self.leakyrelu(x)
        features.append(x)

        x = self.pad(x)
        x = self._conv4(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv5(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv6(x)
        x = self.leakyrelu(x)
        features.append(x)

        x = self.pad(x)
        x = self._conv7(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv8(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv9(x)
        x = self.leakyrelu(x)
        features.append(x)

        x = self.pad(x)
        x = self._conv10(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv11(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv12(x)
        x = self.leakyrelu(x)
        features.append(x)

        x = self.pad(x)
        x = self._conv13(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv14(x)
        x = self.leakyrelu(x)
        x = self.pad(x)
        x = self._conv15(x)
        x = self.leakyrelu(x)
        features.append(x)

        if split_features_by_sample:
            # Split the list of features per level (for all samples) into a nested
            # list that can be indexed by [sample][level].

            n = len(features[0])
            features1 = []
            for i in range(n):
                feature = []
                for f in features:
                    feature.append(f[i][None])
                features1.append(feature)
            features = features1
        return features
