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

"""Library for loading train and eval data.
This libaray contains two functions, make_train_iterator for generating a
training data iterator from multiple sources of different formats and
make_eval_function for creating an evaluation function that evaluates on
data from multiple sources of different formats.
"""

# pylint:disable=g-importing-member
import glob
import random
import numpy as np
from PIL import Image
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset as ds
from mindspore.dataset.transforms.py_transforms import Compose
from src.config import config
# pylint:disable=g-long-lambda


def create_dataset_train(
        train_on,
        height,
        width,
        shuffle_buffer_size,
        batch_size,
        rank=0,
        group_size=1,
):
    """Build joint training iterator for all data in train_on.
    Args:
      train_on: string of the format 'format0:path0;format1:path1', e.g.
         'kitti:/usr/local/home/...'.
      height: int, height to which the images will be resized or cropped.
      width: int, width to which the images will be resized or cropped.
      shuffle_buffer_size: int, size that will be used for the shuffle buffer.
      batch_size: int, batch size for the iterator.
      seed: A seed for a random number generator, controls shuffling of data.
      mode: str, will be passed on to the data iterator class. Can be used to
        specify different settings within the data iterator.
    Returns:
      A tf.data.Iterator that produces batches of images of shape [batch
      size, sequence length=3, height, width, channels=3]
    """
    data_format, path = train_on.split(':')

    file = glob.glob(path + '/*mindrecord')
    if group_size == 1:
        mindrecord_dataset = ds.MindDataset(file,
                                            #num_parallel_workers=8,
                                            shuffle=True
                                            )
    else:
        mindrecord_dataset = ds.MindDataset(file,
                                            num_parallel_workers=8,
                                            num_shards=group_size,
                                            shard_id=rank,
                                            shuffle=True
                                            )

    mindrecord_dataset = mindrecord_dataset.shuffle(config.shuffle_buffer_size)
    mindrecord_dataset = mindrecord_dataset.repeat()
    def roll_op(data1, data2):
        r = random.randint(0, 2)
        return np.roll(data1, r, axis=-1), np.roll(data2, r, axis=-1)

    def reverse_color_op(data1, data2):
        r = random.randint(0, 1)
        if r == 1:
            data1 = np.flip(data1, -1)
            data2 = np.flip(data2, -1)
        return data1, data2

    def reverse_height_op(data1, data2):
        r = random.randint(0, 1)
        if r == 1:
            data1 = np.flip(data1, -3)
            data2 = np.flip(data2, -3)
        return data1, data2

    def adjust_hue_op(data1, data2):
        image_hue_factor = random.uniform(-0.5, 0.5)
        data1 = Image.fromarray(data1)
        data2 = Image.fromarray(data2)
        mode = data1.mode
        hue1, saturation1, value1 = data1.convert('HSV').split()
        hue2, saturation2, value2 = data2.convert('HSV').split()
        np_hue1 = np.array(hue1, dtype=np.uint8)
        np_hue2 = np.array(hue2, dtype=np.uint8)

        with np.errstate(over='ignore'):
            np_hue1 += np.uint8(image_hue_factor * 255)
            np_hue2 += np.uint8(image_hue_factor * 255)
        hue1 = Image.fromarray(np_hue1, 'L')
        hue2 = Image.fromarray(np_hue2, 'L')

        image1 = Image.merge('HSV', (hue1, saturation1, value1)).convert(mode)
        image2 = Image.merge('HSV', (hue2, saturation2, value2)).convert(mode)
        return image1, image2

    def copy_op(data1, data2):
        data2 = data1
        return data1, data2

    def stack_op(data1, data2):
        return np.stack((data1, data2), axis=0), np.stack((data1, data2), axis=0)

    if 'sintel' in data_format:
        transforms_list1 = Compose([py_vision.Decode(),
                                    py_vision.Resize([height, width])])

        transforms_list3 = [roll_op,
                            reverse_color_op]

        #将原始图片1和2解码并resize
        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list1, input_columns=["images1_data"],
                                                    output_columns=["image1_without_photo_aug"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list1, input_columns=["images2_data"],
                                                    output_columns=["image2_without_photo_aug"],
                                                    num_parallel_workers=8)
        #将原始图片1和2进行随机垂直反转
        mindrecord_dataset = mindrecord_dataset.map(operations=reverse_height_op,
                                                    input_columns=["image1_without_photo_aug",
                                                                   "image2_without_photo_aug"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=reverse_height_op,
                                                    input_columns=["image1_without_photo_aug",
                                                                   "image2_without_photo_aug"],
                                                    num_parallel_workers=8)

        #复制图片1和2给准备增强的图片
        mindrecord_dataset = mindrecord_dataset.map(operations=copy_op,
                                                    input_columns=["image1_without_photo_aug", "images1_aug"],
                                                    output_columns=["image1_without_photo_aug", "image1"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=copy_op,
                                                    input_columns=["image2_without_photo_aug", "images2_aug"],
                                                    output_columns=["image2_without_photo_aug", "image2"],
                                                    num_parallel_workers=8)

        #图片1和2 totensor()
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToTensor(),
                                                    input_columns=["image1_without_photo_aug"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToTensor(),
                                                    input_columns=["image2_without_photo_aug"],
                                                    num_parallel_workers=8)

        #增强图片1和2进行增强
        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list3, input_columns=["image1", "image2"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToPIL(), input_columns=["image1"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToPIL(), input_columns=["image2"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=adjust_hue_op, input_columns=["image1", "image2"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToTensor(), input_columns=["image1"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToTensor(), input_columns=["image2"],
                                                    num_parallel_workers=8)

    elif 'chairs' in data_format:
        transforms_list1 = Compose([py_vision.Decode(),
                                    py_vision.Resize([height, width]),
                                    py_vision.ToTensor()])

        transforms_list2 = Compose([py_vision.Decode(),
                                    py_vision.Resize([height, width])])

        transforms_list3 = [roll_op,
                            reverse_color_op]

        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list1, input_columns=["images1_data"],
                                                    output_columns=["image1_without_photo_aug"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list1, input_columns=["images2_data"],
                                                    output_columns=["image2_without_photo_aug"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list2, input_columns=["images1_aug"],
                                                    output_columns=["image1"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list2, input_columns=["images2_aug"],
                                                    output_columns=["image2"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list3, input_columns=["image1", "image2"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToPIL(), input_columns=["image1"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToPIL(), input_columns=["image2"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=adjust_hue_op, input_columns=["image1", "image2"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToTensor(), input_columns=["image1"],
                                                    num_parallel_workers=8)
        mindrecord_dataset = mindrecord_dataset.map(operations=py_vision.ToTensor(), input_columns=["image2"],
                                                    num_parallel_workers=8)
    else:
        print('Unknown data format "{}"'.format(data_format))

    mindrecord_dataset = mindrecord_dataset.map(operations=stack_op,
                                                input_columns=["image1_without_photo_aug", "image2_without_photo_aug"],
                                                output_columns=["images_without_photo_aug",
                                                                "images_without_photo_aug_copy"])

    mindrecord_dataset = mindrecord_dataset.map(operations=stack_op,
                                                input_columns=["image1", "image2"],
                                                output_columns=["images", "images_copy"])

    mindrecord_dataset = mindrecord_dataset.batch(config.batch_size)

    return mindrecord_dataset

def create_dataset_eval(
        eval_on,
        height,
        width,
        shuffle_buffer_size,
        batch_size
):
    """Build joint training iterator for all data in train_on.
    Args:
      eval_on: string of the format 'format0:path0;format1:path1', e.g.
         'kitti:/usr/local/home/...'.
      height: int, height to which the images will be resized or cropped.
      width: int, width to which the images will be resized or cropped.
      shuffle_buffer_size: int, size that will be used for the shuffle buffer.
      batch_size: int, batch size for the iterator.
      seed: A seed for a random number generator, controls shuffling of data.
      mode: str, will be passed on to the data iterator class. Can be used to
        specify different settings within the data iterator.
    Returns:
      A tf.data.Iterator that produces batches of images of shape [batch
      size, sequence length=3, height, width, channels=3]
    """
    data_format, path = eval_on.split(':')

    file = glob.glob(path + '/*mindrecord')

    mindrecord_dataset = ds.MindDataset(file)

    for data in mindrecord_dataset.create_dict_iterator(output_numpy=True):
        origin_height = data["height"].item()
        origin_width = data["width"].item()
        break

    def reshape_flow_op(data):
        return np.reshape(data, (origin_height, origin_width, 2))[Ellipsis, ::-1]

    def flow_valid_op(data1, data2):
        data2 = np.ones_like(data1[:1, Ellipsis])
        return data1, data2

    def reshape_occlusion_op(data):
        return np.reshape(data, (origin_height, origin_width, 1))

    def transpose_op(data):
        transpose_x = np.transpose(data, axes=[2, 0, 1])
        return transpose_x

    def stack_op(data1, data2):
        return np.stack((data1, data2), axis=0), np.stack((data1, data2), axis=0)

    transforms_list1 = Compose([py_vision.Decode(),
                                py_vision.ToTensor()])

    transforms_flow = [reshape_flow_op,
                       transpose_op]

    transforms_occlusion = [reshape_occlusion_op,
                            transpose_op]

    mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list1, input_columns=["images1_data"])
    mindrecord_dataset = mindrecord_dataset.map(operations=transforms_list1, input_columns=["images2_data"])
    mindrecord_dataset = mindrecord_dataset.map(operations=transforms_flow, input_columns=["flow_uv"])
    mindrecord_dataset = mindrecord_dataset.map(operations=flow_valid_op, input_columns=["flow_uv", "flow_path"],
                                                output_columns=["flow_uv", "flow_valid"])
    if 'sintel' in data_format:
        mindrecord_dataset = mindrecord_dataset.map(operations=transforms_occlusion, input_columns=["occlusion_mask"])

    mindrecord_dataset = mindrecord_dataset.map(operations=stack_op,
                                                input_columns=["images1_data", "images2_data"],
                                                output_columns=["images", "images_copy"])

    mindrecord_dataset = mindrecord_dataset.batch(config.batch_size)
    mindrecord_dataset = mindrecord_dataset.shuffle(config.shuffle_buffer_size)

    return mindrecord_dataset
