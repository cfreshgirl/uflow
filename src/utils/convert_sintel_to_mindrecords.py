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

"""This script converts Sintel data to the MindRecords format."""


import os
import glob
import logging
import shutil
from io import BytesIO
from absl import app
from absl import flags
from PIL import Image
import numpy as np
from src.utils import conversion_utils
from mindspore.mindrecord import FileWriter

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'Dataset folder.')
flags.DEFINE_string('output_dir', '', 'Location to export to.')
flags.DEFINE_integer('shard', 0, 'Which shard this is.')
flags.DEFINE_integer('num_shards', 100, 'How many total shards there are.')

#print log
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

def convert_dataset():
    """Convert the data to the TFRecord format."""

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    for data_split in ['training', 'test']:
        split_folder = os.path.join(FLAGS.output_dir, data_split)
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)

    for data_type in ['clean', 'final']:
        output_folder = os.path.join(FLAGS.output_dir, data_split, data_type)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder = os.path.join(FLAGS.output_dir, data_split, data_type)
        input_folder = os.path.join(FLAGS.data_dir, data_split, data_type)
        flow_folder = os.path.join(FLAGS.data_dir, data_split, 'flow')
        occlusion_folder = os.path.join(FLAGS.data_dir, data_split, 'occlusions')
        invalid_folder = os.path.join(FLAGS.data_dir, data_split, 'invalid')

        # Directory with images.
        image_folders = sorted(glob.glob(input_folder + '/*'))

        if data_split == 'training':
            occlusion_folders = sorted(glob.glob(occlusion_folder + '/*'))
            invalid_folders = sorted(glob.glob(invalid_folder + '/*'))
            flow_folders = sorted(glob.glob(flow_folder + '/*'))
            assert len(image_folders) == len(flow_folders)
            assert len(flow_folders) == len(invalid_folders)
            assert len(invalid_folders) == len(occlusion_folders)
        else:  # Test has no ground truth flow.
            flow_folders = occlusion_folders = invalid_folders = [
                None for _ in image_folders
            ]

        data_list = []
        for image_folder, flow_folder, occlusion_folder, invalid_folder in zip(
                image_folders, flow_folders, occlusion_folders, invalid_folders):
            images = glob.glob(os.path.join(image_folder, '*.png'))
            # We may want to eventually look at sequences of frames longer than 2.
            # pylint:disable=g-long-lambda
            sort_by_frame_index = lambda x: int(
                os.path.basename(x).split('_')[1].split('.')[0])
            images = sorted(images, key=sort_by_frame_index)

        if data_split == 'training':
            flows = glob.glob(os.path.join(flow_folder, '*.flo'))
            flows = sorted(flows, key=sort_by_frame_index)
            occlusions = glob.glob(os.path.join(occlusion_folder, '*.png'))
            occlusions = sorted(occlusions, key=sort_by_frame_index)
            invalids = glob.glob(os.path.join(invalid_folder, '*.png'))
            invalids = sorted(invalids, key=sort_by_frame_index)
        else:  # Test has no ground truth flow.
            flows = occlusions = [None for _ in range(len(images) - 1)]
            invalids = [None for _ in images]

        image_pairs = zip(images[:-1], images[1:])
        invalid_pairs = zip(invalids[:-1], invalids[1:])
        # there should be 1 fewer flow images than video frames
        assert len(flows) == len(images) - 1 == len(occlusions)
        assert len(invalids) == len(images)
        data_list.extend(zip(image_pairs, flows, occlusions, invalid_pairs))

    write_records(data_list, output_folder, data_split)


def write_records(data_list, output_folder, data_split):
    """Takes in list: [((im1_path, im2_path), flow_path)] and writes records."""

    if output_folder and not os.path.exists(output_folder):
        print('Making new checkpoint directory', output_folder)
        os.mkdir(output_folder)

    # Reading ppm and flo can fail on network filesystem, so copy to tmpdir first.

    filenames = conversion_utils.generate_sharded_filenames_new(
        os.path.join(output_folder, 'sintel@{}'.format(FLAGS.num_shards)))
    total = len(data_list)
    images_per_shard = total // FLAGS.num_shards
    start = images_per_shard * FLAGS.shard
    end = start + images_per_shard
    # Account for num images not being divisible by num shards.
    if FLAGS.shard == FLAGS.num_shards - 1:
        data_list = data_list[start:]
    else:
        data_list = data_list[start:end]

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Writing %d images per shard', images_per_shard)
    logger.info('Writing range %d to %d of %d total.', start, end, total)

    schema_json = {
        "height": {"type": "int64"},
        "width": {"type": "int64"},
        "images1_aug": {"type": "bytes"},
        "images2_aug": {"type": "bytes"},
        "images1_data": {"type": "bytes"},
        "images2_data": {"type": "bytes"},
    }
    if data_split == 'training':
        schema_json.update({
            "invalid1_data": {"type": "bytes"},
            "invalid2_data": {"type": "bytes"},
            "flow_uv": {"type": "float32", "shape": [-1]},
            "occlusion_mask": {"type": "bytes"},
            "flow_path": {"type": "bytes"},
            "occlusion_path": {"type": "bytes"},
        })

    write(filenames, schema_json, data_list)

    logger.info('Saved results to %s', FLAGS.output_dir)

def write(filenames, schema_json, data_list):

    tmpdir = '/tmp/flying_chairs'
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    img1_path = os.path.join(tmpdir, 'img1.png')
    img2_path = os.path.join(tmpdir, 'img2.png')
    flow_path = os.path.join(tmpdir, 'flow.flo')
    occlusion_path = os.path.join(tmpdir, 'occlusion.png')
    invalid1_path = os.path.join(tmpdir, 'invalid1.png')
    invalid2_path = os.path.join(tmpdir, 'invalid2.png')

    writer = FileWriter(filenames[FLAGS.shard], shard_num=1)
    writer.add_schema(schema_json, "test_schema")

    for i, (images, flow, occlusion, invalids) in enumerate(data_list):
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)
        if os.path.exists(flow_path):
            os.remove(flow_path)
        if os.path.exists(occlusion_path):
            os.remove(occlusion_path)
        if os.path.exists(invalid1_path):
            os.remove(invalid1_path)
        if os.path.exists(invalid2_path):
            os.remove(invalid2_path)

        shutil.copy(images[0], img1_path)
        shutil.copy(images[1], img2_path)

        bytesIO1 = BytesIO()
        bytesIO2 = BytesIO()
        image1_data = Image.open(img1_path)
        image1_data.save(bytesIO1, format='PNG')
        image2_data = Image.open(img2_path)
        image2_data.save(bytesIO2, format='PNG')

        if flow is not None:
            assert occlusion is not None
            shutil.copy(flow, flow_path)
            shutil.copy(occlusion, occlusion_path)
            shutil.copy(invalids[0], invalid1_path)
            shutil.copy(invalids[1], invalid2_path)
            flow_data = conversion_utils.read_flow(flow_path)
            occlusion_data = np.expand_dims(
                np.asarray(Image.open(occlusion_path)) // 255, axis=-1)
            invalid1_data = np.expand_dims(
                np.asarray(Image.open(invalid1_path)) // 255, axis=-1)
            invalid2_data = np.expand_dims(
                np.asarray(Image.open(invalid2_path)) // 255, axis=-1)
        else:  # Test has no flow data, spoof flow data.
            flow_data = np.zeros((image1_data.size[1], image1_data.size[0], 2),
                                 np.float32)
            occlusion_data = invalid1_data = invalid2_data = np.zeros(
                (image1_data.size[1], image1_data.size[0], 1), np.uint8)

        height = image1_data.size[1]
        width = image1_data.size[0]

        assert height == image2_data.size[1] == flow_data.shape[0]
        assert width == image2_data.size[0] == flow_data.shape[1]
        assert height == occlusion_data.shape[0] == invalid1_data.shape[0]
        assert width == occlusion_data.shape[1] == invalid1_data.shape[1]
        assert invalid1_data.shape == invalid2_data.shape

        data = [{
            "height": height,
            "width": width,
            "image1_path": str.encode(images[0]),
            "image2_path": str.encode(images[1]),
            "images1_aug": bytesIO1.getvalue(),
            "images2_aug": bytesIO2.getvalue(),
            "images1_data": bytesIO1.getvalue(),
            "images2_data": bytesIO2.getvalue(),
        }]

        if flow is not None:
            data[0].update({
                "invalid1_data": invalid1_data.tobytes(),
                "invalid2_data": invalid2_data.tobytes(),
                "flow_uv": np.reshape(flow_data, (-1)),
                "occlusion_mask": occlusion_data.tobytes(),
                "flow_path": str.encode(flow),
                "occlusion_path": str.encode(occlusion)
            })

        writer.write_raw_data(data)
        if i % 10 == 0:
            logger.info('Writing %d out of %d total.', i,
                        len(data_list))

    writer.commit()

def main(_):
    convert_dataset()


if __name__ == '__main__':
    app.run(main)
