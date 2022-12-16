"""Continually polls and evaluates new checkpoints."""
import time
import glob
import ast
import argparse
import os
import numpy as np
from mindspore import context, ops
import mindspore
from src.logger import get_logger
from src.dataset import create_dataset_eval
from src.Uflow import PWCFlow, PWCFeaturePyramid
from src.infer import InferNet
from src.config import config

parser = argparse.ArgumentParser(description='What Matters In Unsupervised Optical Flow')

parser.add_argument('--eval_on', type=str, default='', help='"format0:path0", e.g. "flyingchairs:/usr/..."')
parser.add_argument('--checkpoint_dir', type=str, default='', help='Path to directory for saving checkpoints.')
parser.add_argument('--height', type=int, default=448, help='Image height for training and evaluation.')
parser.add_argument('--width', type=int, default=1024, help='Image width for training and evaluation.')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--isModelArts', type=ast.literal_eval, default=False)
parser.add_argument('--distribute', type=ast.literal_eval, default=False)
parser.add_argument('--device_id', type=int, default=1, help='device_id')

args = parser.parse_args()

def make_eval_function(feature_model, flow_model):
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

    return infer_net

def main():

    target = args.device_target
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    # Make directories if they do not exist yet.
    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        print('Making new checkpoint directory', args.checkpoint_dir)
        os.makedirs(args.checkpoint_dir)

    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    logger = get_logger("./", 0)
    logger.save_args(config)

    #define net
    flow_model = PWCFlow(
        dropout_rate=0.1,
        normalize_before_cost_volume=True,
        num_levels=5,
        use_feature_warp=True,
        use_cost_volume=True,
        channel_multiplier=1,
        accumulate_flow=True,
        shared_flow_decoder=False)
    feature_model = PWCFeaturePyramid(
        level1_num_layers=3,
        level1_num_filters=32,
        level1_num_1x1=0,
        original_layer_sizes=False,
        num_levels=5,
        channel_multiplier=1)

    dataset = create_dataset_eval(args.eval_on,
                                  args.height,
                                  args.width,
                                  config.shuffle_buffer_size,
                                  config.batch_size)

    feature_model.set_train(False)
    flow_model.set_train(False)

    infer_net = make_eval_function(feature_model, flow_model)

    checkpoints = sorted(glob.glob(args.checkpoint_dir + '/*.ckpt'))

    for ckpt in checkpoints:
        current_epoch = int(ckpt.split("uflow_")[1].split("_")[0])
        if current_epoch < 800:
            continue
        param_dict = mindspore.load_checkpoint(ckpt)
        mindspore.load_param_into_net(feature_model, param_dict)
        mindspore.load_param_into_net(flow_model, param_dict)

        data_format = args.eval_on.split(":")[0]
        has_occlusion = True
        if 'flyingchairs' in args.eval_on:
            has_occlusion = False

        eval_start_in_s = time.time()
        epe_occ = []  # End point errors.
        errors_occ = []
        for test_batch in dataset.create_dict_iterator():

            if has_occlusion:
                flow_gt = test_batch['flow_uv'][0]
                image_batch = test_batch['images'][0]
                occ_mask_gt = test_batch['occlusion_mask'][0]
            else:
                flow_gt = test_batch['flow_uv'][0]
                image_batch = test_batch['images'][0]
                occ_mask_gt = ops.OnesLike()(flow_gt[-1:, Ellipsis])

            endpoint_error_occ, outliers_occ = infer_net(image_batch[0],
                                                         image_batch[1],
                                                         flow_gt,
                                                         occ_mask_gt,
                                                         input_height=args.height,
                                                         input_width=args.width,
                                                         resize_flow_to_img_res=True,
                                                         has_occlusion=has_occlusion,
                                                         infer_occlusion=True)

            epe_occ.append(np.mean(endpoint_error_occ.asnumpy()))
            errors_occ.append(np.mean(outliers_occ.asnumpy()))

        eval_end_in_s = time.time()
        results = {
            data_format+'epoch': current_epoch,
            data_format+'EPE': np.mean(np.array(epe_occ)),
            data_format+'ER': np.mean(np.array(errors_occ)),
            data_format+'eval-time(s)': eval_end_in_s - eval_start_in_s
        }

        logger.info(results)


if __name__ == '__main__':
    main()
