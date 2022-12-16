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

"""Main script to train and evaluate UFlow."""

# pylint:disable=g-importing-member
import os
import time
import ast
import argparse
from mindspore import Tensor, nn, context, load_checkpoint, \
    load_param_into_net, save_checkpoint, dataset as ds, dtype as mstype
from mindspore.communication.management import init, get_group_size, get_rank
# pylint:disable=unused-import
from src.dataset import create_dataset_train, create_dataset_eval
from src.config import config
from src.logger import get_logger
from src.Uflow import PWCFlow, PWCFeaturePyramid
from src.network_with_loss import UflowLoss, UflowNetWithLoss
from src.infer import InferNet

parser = argparse.ArgumentParser(description='What Matters In Unsupervised Optical Flow')

parser.add_argument('--train_url', type=str)
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_on', type=str, default='', help='"format0:path0;format1:path1", e.g. "kitti:/usr/..."')
parser.add_argument('--eval_on', type=str, default='', help='"format0:path0;format1:path1", e.g. "kitti:/usr/..."')
parser.add_argument('--pre_trained', type=str, default=False)
parser.add_argument('--pre_ckpt_path', type=str, default='', help='Pretrain path')
parser.add_argument('--checkpoint_dir', type=str, default='', help='Path to directory for saving checkpoints.')
parser.add_argument('--height', type=int, default=384, help='Image height for training and evaluation.')
parser.add_argument('--width', type=int, default=512, help='Image width for training and evaluation.')
parser.add_argument('--lr', type=float, default=0.0001, help='learning_rate.')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--isModelArts', type=ast.literal_eval, default=False)
parser.add_argument('--distribute', type=ast.literal_eval, default=False)
parser.add_argument('--device_id', type=int, default=6, help='device_id')

args = parser.parse_args()

if args.isModelArts:
    import moxing as mox

#set_seed(41)
ds.config.set_seed(41)

if __name__ == '__main__':

    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    # Make directories if they do not exist yet.
    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        print('Making new checkpoint directory', args.checkpoint_dir)
        os.makedirs(args.checkpoint_dir)

    if args.distribute:
        if target == "Ascend":
            init()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              parameter_broadcast=True)
        if target == "GPU":
            init()
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        if target == "Ascend":
            rank = 0
            device_num = 1
            device_id = args.device_id
            context.set_context(device_id=args.device_id)

    logger = get_logger("./", rank)
    logger.save_args(config)

    if args.isModelArts:
        import moxing as mox

        # download dataset from obs to cache
        mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        data_format, path = args.train_on.split(':')
        train_dataset_path = data_format + ':/cache/dataset/device_' + os.getenv('DEVICE_ID') + path
        # create dataset
        # Build training iterator.
        logger.info('Making training iterator.')
        train_dataset = create_dataset_train(
            train_dataset_path,
            args.height,
            args.width,
            config.shuffle_buffer_size,
            config.batch_size,
            rank=rank,
            group_size=device_num)
        if args.eval_on:
            data_format, path = args.eval_on.split(':')
            eval_dataset_path = data_format + ':/cache/dataset/device_' + os.getenv('DEVICE_ID') + path
            # create dataset
            # Build training iterator.
            logger.info('Making training iterator.')
            eval_dataset = create_dataset_eval(eval_dataset_path,
                                               args.height,
                                               args.width,
                                               config.shuffle_buffer_size,
                                               config.batch_size)
    else:
        # Build training iterator.
        logger.info('Making training iterator.')
        train_dataset = create_dataset_train(
            args.train_on,
            args.height,
            args.width,
            config.shuffle_buffer_size,
            config.batch_size,
            rank=rank,
            group_size=device_num)
        if args.eval_on:
            eval_dataset = create_dataset_eval(args.eval_on,
                                               args.height,
                                               args.width,
                                               config.shuffle_buffer_size,
                                               config.batch_size)

    train_dataset = train_dataset.create_dict_iterator()
    step_size = config.epoch_length//device_num

    # define net
    feature_model = PWCFeaturePyramid(
        level1_num_layers=config.level1_num_layers,
        level1_num_filters=config.level1_num_filters,
        level1_num_1x1=config.level1_num_1x1,
        original_layer_sizes=config.original_layer_sizes,
        num_levels=config.num_levels,
        channel_multiplier=config.channel_multiplier)
    flow_model = PWCFlow(
        dropout_rate=config.dropout_rate,
        normalize_before_cost_volume=config.normalize_before_cost_volume,
        num_levels=config.num_levels,
        use_feature_warp=config.use_feature_warp,
        use_cost_volume=config.use_cost_volume,
        channel_multiplier=config.channel_multiplier,
        accumulate_flow=config.accumulate_flow,
        shared_flow_decoder=config.shared_flow_decoder)

    if args.eval_on:
        occ_weights = {
            'fb_abs': config.occ_weights_fb_abs,
            'forward_collision': config.occ_weights_forward_collision,
            'backward_zero': config.occ_weights_backward_zero,
        }
        # Switch off loss-terms that have weights < 1e-2.
        occ_weights = {k: v for (k, v) in occ_weights.items() if v > 1e-2}

        occ_thresholds = {
            'fb_abs': config.occ_thresholds_fb_abs,
            'forward_collision': config.occ_thresholds_forward_collision,
            'backward_zero': config.occ_thresholds_backward_zero,
        }

        occ_clip_max = {
            'fb_abs': config.occ_clip_max_fb_abs,
            'forward_collision': config.occ_clip_max_forward_collision,
        }

        infer_net = InferNet(
            feature_model=feature_model,
            flow_model=flow_model,
            checkpoint_dir=args.checkpoint_dir,
            num_levels=5,
            occlusion_estimation=config.occlusion_estimation,
            occ_weights=occ_weights,
            occ_thresholds=occ_thresholds,
            occ_clip_max=occ_clip_max)

        infer_net.set_train(False)

    current_epoch = 0
    if args.pre_trained:
        param_dict = load_checkpoint(args.pre_ckpt_path)
        logger.info(args.pre_ckpt_path)
        current_epoch = int(args.pre_ckpt_path.split("uflow_")[1].split("_")[0])
        if current_epoch >= config.num_train_steps//config.epoch_length:
            current_epoch = 0
        load_param_into_net(feature_model, param_dict)
        load_param_into_net(flow_model, param_dict)

    logger.info("----------define lossnet-----------")
    uflow_loss = UflowLoss(
        fb_sigma_teacher=config.fb_sigma_teacher,
        fb_sigma_student=config.fb_sigma_student,
        smoothness_edge_weighting=config.smoothness_edge_weighting,
        stop_gradient_mask=config.stop_gradient_mask,
        selfsup_mask=config.selfsup_mask,
        smoothness_at_level=config.smoothness_at_level,
        smooth1_weight=config.weight_smooth1,
        census_weight=config.weight_census,
        smoothness_edge_constant=config.smoothness_edge_constant)

    net_with_loss = UflowNetWithLoss(feature_model, flow_model, uflow_loss)

    params_all = feature_model.trainable_params() + flow_model.trainable_params()

    net_opt = nn.Adam(params=params_all, learning_rate=args.lr)

    network = nn.TrainOneStepCell(net_with_loss, net_opt)
    #network.set_train()
    t_end = time.time()

    def weight_selfsup_fn(epochidx):
        step = (epochidx * config.epoch_length) % config.selfsup_step_cycle

        # Start self-supervision only after a certain number of steps.
        # Linearly increase self-supervision weight for a number of steps.

        ramp_up_factor = (step - (config.selfsup_after_num_steps - 1)) / config.selfsup_ramp_up_steps
        x_min = min(ramp_up_factor, 1.)
        x_max = max(x_min, 0.)
        return config.weight_selfsup * x_max

    logger.info(current_epoch)
    logger.info('==========start training===============')

    for epoch_idx in range(current_epoch, config.num_train_steps//config.epoch_length):
        # Set which occlusion estimation methods could be active at this point.
        # (They will only be used if occlusion_estimation is set accordingly.)
        occ_active = {
            'uflow':
                config.occlusion_estimation == 'uflow',
            'brox':
                epoch_idx * config.epoch_length > config.occ_after_num_steps_brox,
            'wang':
                epoch_idx * config.epoch_length > config.occ_after_num_steps_wang,
            'wang4':
                epoch_idx * config.epoch_length > config.occ_after_num_steps_wang,
            'wangthres':
                epoch_idx * config.epoch_length > config.occ_after_num_steps_wang,
            'wang4thres':
                epoch_idx * config.epoch_length > config.occ_after_num_steps_wang,
            'fb_abs':
                epoch_idx * config.epoch_length > config.occ_after_num_steps_fb_abs,
            'forward_collision':
                epoch_idx * config.epoch_length > config.occ_after_num_steps_forward_collision,
            'backward_zero':
                epoch_idx * config.epoch_length > config.occ_after_num_steps_backward_zero,
        }

        # Prepare self-supervision if it will be used in the next epoch.
        if config.weight_selfsup > 1e-7 and (
                epoch_idx % config.selfsup_step_cycle) >= 500:

            # Add selfsup weight with a ramp-up schedule. This will cause a
            # recompilation of the training graph defined in uflow.train(...).
            selfsup_weight = Tensor(weight_selfsup_fn(epoch_idx), mstype.float32)
        else:
            selfsup_weight = Tensor(weight_selfsup_fn(0), mstype.float32)

        t_end = time.time()
        loss = 0

        for step_idx, data in zip(range(step_size), train_dataset):
            if step_idx >= step_size:
                break
            loss += network(data["images"], data["images_without_photo_aug"], selfsup_weight, occ_active)
        time_used = (time.time() - t_end) * 1000
        loss = loss / step_size
        logger.info('epoch %d , loss %f, per step time: %f ms', epoch_idx + 1, loss, time_used/step_size)

        if args.isModelArts:
            if not os.path.exists('/cache/outputs/device_' + os.getenv('DEVICE_ID') + '/'):
                os.mkdir('/cache/outputs/device_' + os.getenv('DEVICE_ID') + '/')
            save_checkpoint_path = '/cache/outputs/device_' + os.getenv('DEVICE_ID') + '/'
        else:
            if target == "GPU" and args.distribute:
                save_checkpoint_path = os.path.join(args.checkpoint_dir, 'ckpt_' + str(get_rank()) + '/')
            else:
                save_checkpoint_path = args.checkpoint_dir

        if (epoch_idx + 1) % 10 == 0 or epoch_idx >= 800:
            if args.distribute:
                if get_rank() == 0:
                    ckpt_name = os.path.join(save_checkpoint_path,
                                             "uflow_{}_{}.ckpt".format(epoch_idx + 1, config.epoch_length))
                    save_checkpoint(network, ckpt_name)
            else:
                ckpt_name = os.path.join(save_checkpoint_path,
                                         "uflow_{}_{}.ckpt".format(epoch_idx + 1, config.epoch_length))
                save_checkpoint(network, ckpt_name)
    # loss = net_with_loss(images_aug, images_without_aug, weights)
    if args.isModelArts:
        mox.file.copy_parallel(src_url='/cache/outputs', dst_url=args.train_url)

    logger.info('==========end training===============')
