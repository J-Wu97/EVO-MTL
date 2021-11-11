import os
import numpy as np
from glob import glob
import imageio

import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_dir + data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks


class MultiTaskDataset(Dataset):
    """MultiTaskDataset."""

    def __init__(self, data_dir, image_mean, data_list_1, data_list_2, output_size,
                 color_jitter, random_scale, random_mirror, random_crop, ignore_label):
        """
        Initialise an Multitask Dataloader.

        :param data_dir: path to the directory with images and masks.
        :param data_list_1: path to the file with lines of the form '/path/to/image /path/to/mask'.
        :param data_list_2: path to the file with lines of the form '/path/to/image /path/to/mask'.
        :param output_size: a tuple with (height, width) values, to which all the images will be resized to.
        :param random_scale: whether to randomly scale the images.
        :param random_mirror: whether to randomly mirror the images.
        :param random_crop: whether to randomly crop the images.
        :param ignore_label: index of label to ignore during the training.
        """
        # assert dataset == '../nyu_v2'

        self.data_dir = data_dir
        self.image_mean = np.load(self.data_dir + image_mean)
        self.data_list_1 = data_list_1
        self.data_list_2 = data_list_2
        self.output_size = output_size

        self.color_jitter = None
        if color_jitter:
            print("Using color jitter")
            self.color_jitter = transforms.ColorJitter(hue=.05, saturation=.05)
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.random_crop = random_crop

        self.ignore_label = ignore_label

        image_list_1, self.label_list_1 = read_labeled_image_list(self.data_dir, self.data_list_1)  # 得到图像和标签图像地址
        image_list_2, self.label_list_2 = read_labeled_image_list(self.data_dir, self.data_list_2)
        assert (image_list_1 == image_list_2)
        self.image_list = image_list_1

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.image_mean, (1., 1., 1.))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label_1 = Image.open(self.label_list_1[idx])
        label_2 = Image.open(self.label_list_2[idx])
        w, h = image.size

        if self.color_jitter:
            image = self.color_jitter(image)

        if self.random_scale:
            scale = int(min(w, h) * (np.random.uniform() + 0.5))
            resize_bl = transforms.Resize(size=scale, interpolation=PIL.Image.BILINEAR)
            resize_nn = transforms.Resize(size=scale, interpolation=PIL.Image.NEAREST)
            image = resize_bl(image)
            label_1 = resize_nn(label_1)
            label_2 = resize_nn(label_2)

        if self.random_mirror:
            if np.random.uniform() < 0.5:
                image = TF.hflip(image)
                label_1 = TF.hflip(label_1)
                label_2 = TF.hflip(label_2)

        if self.random_crop:
            # pad the width if needed
            if image.size[0] < self.output_size[1]:
                image = TF.pad(image, (self.output_size[1] - image.size[0], 0))
                label_1 = TF.pad(label_1, (self.output_size[1] - label_1.size[0], 0), self.ignore_label, 'constant')
                label_2 = TF.pad(label_2, (self.output_size[1] - label_2.size[0], 0),
                                 tuple([self.ignore_label] * 3), 'constant')
            # pad the height if needed
            if image.size[1] < self.output_size[0]:
                image = TF.pad(image, (0, self.output_size[0] - image.size[1]))
                label_1 = TF.pad(label_1, (0, self.output_size[0] - label_1.size[1]), self.ignore_label, 'constant')
                label_2 = TF.pad(label_2, (0, self.output_size[0] - label_2.size[1]),
                                 tuple([self.ignore_label] * 3), 'constant')

            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.output_size)
            image = TF.crop(image, i, j, h, w)
            label_1 = TF.crop(label_1, i, j, h, w)
            label_2 = TF.crop(label_2, i, j, h, w)

        image = self.normalize(self.to_tensor(np.array(image) - 255.).float() + 255.)
        label_1 = self.to_tensor(np.array(label_1) - 255.) + 255.
        label_2 = self.to_tensor(np.array(label_2) - 255.) + 255.

        return image, label_1.long(), label_2.float()


if __name__ == "__main__":
    data_dir = 'E://MTL//GA-MTL//datasets//nyu_v2'
    image_mean = '//nyu_v2_mean.npy'
    data_list_1 = '//list//training_seg.txt'
    data_list_2 = '//list//training_normal_mask.txt'
    output_size = (321, 321)
    color_jitter = True
    random_scale = True
    random_mirror = True
    random_crop = True
    ignore_label = 255
    data = MultiTaskDataset(data_dir, image_mean, data_list_1, data_list_2, output_size, color_jitter,
                            random_scale, random_mirror, random_crop, ignore_label)

    print(len(data))
    images, label1s, label2s = data[0]
    print(images.size(), label1s.size(), label2s.size())
    num_data = len(data)
    indices = list(range(num_data))
    split = int(np.floor(0.5 * num_data))  # 训练数据的划分
    train_data = torch.utils.data.Subset(data, indices[:split])
    val_data = torch.utils.data.Subset(data, indices[split:num_data])
    print("this is len_train_data", len(train_data))
    print("this is len_val_data", len(train_data))
    # train_loader 实际上是train_data*batch_size进行了一个打包
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=10,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=10,
        pin_memory=True
    )
    print(len(train_loader))  # 长度为40 划分为了40组
    val_iter = iter(val_loader)  # 转换为可以迭代的形式
    for batch_idx, (image, label_1, label_2) in enumerate(train_loader):
        if torch.cuda.is_available():
            image, label_1, label_2 = image.cuda(), label_1.cuda(), label_2.cuda()

        # get a random minibatch from the search queue without replacement
        val_batch = next(val_iter, None)
        if val_batch is None:  # val_iter has reached its end
            val_iter = iter(val_loader)
            val_batch = next(val_iter)
        image_search, label_1_search, label_2_search = val_batch
        if torch.cuda.is_available():
            image_search = image_search.cuda()
            label_1_search, label_2_search = label_1_search.cuda(), label_2_search.cuda()

    #     # setting flag for training arch parameters
    #     model.arch_train()
    #     assert model.arch_training
    #     arch_optimizer.zero_grad()
    #     arch_result = model.loss(image_search, (label_1_search, label_2_search))
    #     arch_loss = arch_result.loss
    # print(len(train_loader))
    # train_iter = iter(train_loader)
    # train_batch = next(train_iter)
    # image, label1, label2 = train_batch
    # print(image.size(), label1.size(), label2.size())
    # for batch_idx, (image, label_1, label_2) in enumerate(train_loader):
    #     image, label_1, label_2 = image, label_1, label_2
    #     # setting flag for training arch parameters
    #     print(image.size())


