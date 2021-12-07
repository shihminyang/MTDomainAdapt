import cv2
import math
import os
from os.path import join as PJ
import sys
import time
import yaml
import torch
import torch.nn.init as init
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision.utils as vutils

# Methods
# config:                       Loading config file(.yaml).

# dataset_info:                 Getting dataset info to set dataset.
# data_loader:                  Create a data loader.
# translate_data_loaders:       Data loader interface. (for pure style translate)
# all_data_loader:              Primary data loader interface.

# show_image:                   Display image(torch tensor) by cv2.
# draw_detection:               Draw detection result in the image.
# calculate_bounding_boxes:     Calucalate bounding boxes from boxes and boxes
#                               offset and trans back to original images.

# Method from MUNIT(https://github.com/NVlabs/MUNIT)
# weights_init:
# get_model_list:
# get_scheduler:


def prepare_folder(config, output_directory):
    """ Creating experiment result folder """
    checkpoint_directory, image_directory = None, None
    seg_directory, detect_directory = None, None

    checkpoint_directory = PJ(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print(f"Creating directory: {checkpoint_directory}")
        os.makedirs(checkpoint_directory)

    if config['translator'] is not None:
        image_directory = PJ(output_directory, 'translate_images')
        if not os.path.exists(image_directory):
            print(f"Creating directory: {image_directory}")
            os.makedirs(image_directory)

    if config['segmentor'] is not None:
        seg_directory = PJ(output_directory, 'segmentation_image')
        if not os.path.exists(seg_directory):
            print(f"Creating directory: {seg_directory}\n")
            os.makedirs(seg_directory)

    return checkpoint_directory, image_directory, seg_directory


def transform_setup(config, task, mode="train"):
    train = True if mode == "train" else False

    if task == "classification":
        crop, crop_h, crop_w = config["crop"][task]
        new_size = config['new_size'][task]

        transform_list = [transforms.ToTensor(),
                          # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        transform_list = [transforms.CenterCrop((crop_h, crop_w))] + transform_list if crop else transform_list
        transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
        transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
        transform = transforms.Compose(transform_list)

    elif task == "segmentation":
        crop, crop_h, crop_w = config["crop"][task]
        new_size = config['new_size'][task]

        transform_list = {}
        if new_size:
            transform_list["resize"] = {"size": new_size}
        if crop:
            transform_list["random_crop"] = {"size": (crop_h, crop_w)}
        if train:
            transform_list["random_horizontal_flip"] = 1
        transform = transform_list

    elif task == "detection":
        sys.exit('Not support yet...')
    else:
        sys.exit('Task not support!')

    return transform


def calculate_parameters(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        count = 1
        for j in i.size():
            count *= j
        k = k + count
    print(f"Number of total parameters: {str(k)}")


def config(config_path):
    """ Loading config file. """
    with open(config_path, 'r') as f:
        return yaml.load(f)


##################################################
# Code from MUNIT (https://github.com/NVlabs/MUNIT)
##################################################
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print(f'initial weight {init_type}')
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_model_list(dirname, key):
    """ Get model list for resume """
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None    # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    elif hyperparameters['lr_policy'] == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=hyperparameters['step_size'],
                                             gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


class Timer:
    def __init__(self, msg, display, scale):
        self.msg = msg
        self.start_time = None
        self.display = display
        self.scale = scale

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.display:
            spent = time.time() - self.start_time
            print(f"{self.msg} {spent:.4f} | Scale: {self.scale}")


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)\
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def __write_images(image_outputs, display_image_num, file_name):
    # expand gray-scale images to 3 channels
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_tensor = (image_tensor * 0.5) + 0.5
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=False)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))
