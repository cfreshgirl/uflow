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
network config setting, will be used in main.py
"""
from easydict import EasyDict

config = EasyDict({
    "shuffle_buffer_size": 1024,
    "batch_size": 1,
    "selfsup_crop_height": 16,
    "selfsup_crop_width": 16,
    "occ_weights_fb_abs": 1000.0,
    "occ_weights_forward_collision": 1000.0,
    "occ_weights_backward_zero": 1000.0,
    "occ_thresholds_fb_abs": 1.5,
    "occ_thresholds_forward_collision": 0.4,
    "occ_thresholds_backward_zero": 0.25,
    "occ_clip_max_fb_abs": 10.0,
    "occ_clip_max_forward_collision": 5.0,
    "level1_num_layers": 3,
    "level1_num_filters": 32,
    "level1_num_1x1": 0,
    "dropout_rate": 0.1,
    "fb_sigma_teacher": 0.003,
    "fb_sigma_student": 0.03,
    "smoothness_edge_weighting": "exponential",
    "stop_gradient_mask": True,
    "selfsup_mask": "gaussian",
    "normalize_before_cost_volume": True,
    "original_layer_sizes": False,
    "shared_flow_decoder": False,
    "channel_multiplier": 1,
    "num_levels": 5,
    "use_cost_volume": True,
    "use_feature_warp": True,
    "accumulate_flow": True,
    "occlusion_estimation": "wang",
    "smoothness_at_level": 2,
    "weight_census": 1.0,
    "weight_smooth1": 4.0,
    "weight_smooth2": 0.0,
    "smoothness_edge_constant": 150.,
    "selfsup_step_cycle": int(1e10),
    "selfsup_after_num_steps": int(5e5),
    "selfsup_ramp_up_steps": int(1e5),
    "weight_selfsup": 0.6,
    "occ_after_num_steps_brox": 0,
    "occ_after_num_steps_wang": 0,
    "occ_after_num_steps_fb_abs": 0,
    "occ_after_num_steps_forward_collision": 0,
    "occ_after_num_steps_backward_zero": 0,
    "num_train_steps": int(1.2e6),
    "epoch_length": 1000
})
