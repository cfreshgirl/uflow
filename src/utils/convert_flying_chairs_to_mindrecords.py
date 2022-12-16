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

"""This script converts Flying Chairs data to the TFRecords format."""

import os
import glob
import logging
import shutil
from io import BytesIO
import numpy as np
from absl import app
from absl import flags
from PIL import Image
from src.utils import conversion_utils
from mindspore.mindrecord import FileWriter

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'Dataset folder.')
flags.DEFINE_string('output_dir', '', 'Location to export to.')
flags.DEFINE_string('train_split_file', 'files/chairs_train_val.txt',
                    'location of the chairs_train_val.txt file')
flags.DEFINE_integer('shard', 0, 'Which shard this is.')
flags.DEFINE_integer('num_shards', 1, 'How many total shards there are.')

# print log
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

def convert_dataset():
    """Convert the data to the TFRecord format."""

    # Make a directory to save the tfrecords to.
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    train_dir = os.path.join(FLAGS.output_dir, 'train')
    test_dir = os.path.join(FLAGS.output_dir, 'test')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Directory with images.
    images = sorted(glob.glob(FLAGS.data_dir + '/*.ppm'))
    flow_list = sorted(glob.glob(FLAGS.data_dir + '/*.flo'))
    assert len(images) // 2 == len(flow_list)
    image_list = []
    for i in range(len(flow_list)):
        im1 = images[2 * i]
        im2 = images[2 * i + 1]
        image_list.append((im1, im2))
    assert len(image_list) == len(flow_list)

    write_records(train_dir, test_dir, image_list, flow_list)

def write_records(train_dir, test_dir, image_list, flow_list):

    # Reading ppm and flo can fail on network filesystem, so copy to tmpdir first.

    train_filenames = conversion_utils.generate_sharded_filenames_new(
        os.path.join(train_dir, 'flying_chairs@{}'.format(FLAGS.num_shards)))
    test_filenames = conversion_utils.generate_sharded_filenames_new(
        os.path.join(test_dir, 'flying_chairs@{}'.format(FLAGS.num_shards)))

    total = len(image_list)
    images_per_shard = total // FLAGS.num_shards
    start = images_per_shard * FLAGS.shard
    filepath = FLAGS.train_split_file
    with open(filepath, mode='r') as f:
        train_val = f.readlines()
        train_val = [int(x.strip()) for x in train_val]
    if FLAGS.shard == FLAGS.num_shards - 1:
        end = len(image_list)
    else:
        end = start + images_per_shard

    assert len(train_val) == len(image_list)
    assert len(flow_list) == len(train_val)
    image_list = image_list[start:end]
    train_val = train_val[start:end]
    flow_list = flow_list[start:end]

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Writing %d images per shard', images_per_shard)
    logger.info('Writing range %d to %d of %d total.', start, end,
                total)

    schema_json = {
        "height": {"type": "int64"},
        "width": {"type": "int64"},
        "flow_uv": {"type": "float32", "shape": [-1]},
        "flow_path": {"type": "bytes"},
        "images1_aug": {"type": "bytes"},
        "images2_aug": {"type": "bytes"},
        "images1_data": {"type": "bytes"},
        "images2_data": {"type": "bytes"},
    }
    write(train_filenames, test_filenames, schema_json, image_list, flow_list, train_val)
    logger.info('Saved results to %s', FLAGS.output_dir)

def write(train_filenames, test_filenames, schema_json, image_list, flow_list, train_val):
    train_record_writer = FileWriter(train_filenames[FLAGS.shard], shard_num=1)
    test_record_writer = FileWriter(test_filenames[FLAGS.shard], shard_num=1)
    train_record_writer.add_schema(schema_json, "train_schema")
    test_record_writer.add_schema(schema_json, "test_schema")

    tmpdir = '/tmp/flying_chairs'
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    img1_path = os.path.join(tmpdir, 'img1.ppm')
    img2_path = os.path.join(tmpdir, 'img2.ppm')
    flow_path = os.path.join(tmpdir, 'flow.flo')

    for i, (images, flow,
            assignment) in enumerate(zip(image_list, flow_list, train_val)):
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)
        if os.path.exists(flow_path):
            os.remove(flow_path)

        shutil.copy(images[0], img1_path)
        shutil.copy(images[1], img2_path)
        shutil.copy(flow, flow_path)

        bytesIO1 = BytesIO()
        bytesIO2 = BytesIO()
        image1_data = Image.open(img1_path)
        image1_data.save(bytesIO1, format='PNG')
        image2_data = Image.open(img2_path)
        image2_data.save(bytesIO2, format='PNG')
        flow_data = conversion_utils.read_flow(flow_path)

        height = image1_data.size[1]
        width = image1_data.size[0]

        assert height == image2_data.size[1] == flow_data.shape[0]
        assert width == image2_data.size[0] == flow_data.shape[1]

        data = [{
            "height": height,
            "width": width,
            "flow_uv": np.reshape(flow_data, (-1)),
            "flow_path": str.encode(flow),
            "images1_aug": bytesIO1.getvalue(),
            "images2_aug": bytesIO2.getvalue(),
            "images1_data": bytesIO1.getvalue(),
            "images2_data": bytesIO2.getvalue(),
        }]
        if i % 10 == 0:
            logger.info('Writing %d out of %d total.', i,
                        len(image_list))
        if assignment == 1:
            train_record_writer.write_raw_data(data)
        elif assignment == 2:
            test_record_writer.write_raw_data(data)
        else:
            assert False, 'There is an error in the chairs_train_val.txt'

    train_record_writer.commit()
    test_record_writer.commit()

def main(_):
    convert_dataset()


if __name__ == '__main__':
    app.run(main)
