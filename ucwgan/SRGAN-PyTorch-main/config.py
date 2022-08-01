# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Image magnification factor
upscale_factor = 10
# Current configuration parameter method
mode = "train_srresnet"
# Experiment name, easy to save weights and log files
exp_name = "SRResNet_baseline"

if mode == "train_srresnet":
    # Dataset address
    train_X = "/user/home/al18709/work/tc_data_flipped/train_X.npy"
    valid_X = "/user/home/al18709/work/tc_data_flipped/valid_X.npy"
    train_y = "/user/home/al18709/work/tc_data_flipped/train_y.npy"
    valid_y = "/user/home/al18709/work/tc_data_flipped/valid_y.npy"
    train_image_dir = "/user/home/al18709/work/tc_data_flipped/train_"
    valid_image_dir = "/user/home/al18709/work/tc_data_flipped/valid_"
    test_lr_image_dir = f"./data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = f"./data/Set5/GTmod12"

    image_size = 100
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_path = "./user/home/al18709/work/SRResNet/SRResNet_x4-ImageNet-2096ee7f.pth.tar"
    pretrained_model_path = False

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 44

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # How many iterations to print the training result
    print_frequency = 200

if mode == "train_srgan":
    # Dataset address
    train_image_dir = "./data/ImageNet/SRGAN/train"
    valid_image_dir = "./data/ImageNet/SRGAN/valid"
    test_lr_image_dir = f"./data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = f"./data/Set5/GTmod12"

    image_size = 96
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_d_model_path = ""
    pretrained_g_model_path = "./user/home/al18709/work/SRResNet/g_best.pth.tar"

    # Incremental training and migration training
    resume_d = ""
    resume_g = ""

    # Total num epochs
    epochs = 9

    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.35"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # Loss function weight
    content_weight = 1.0
    adversarial_weight = 0.001

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # Dynamically adjust the learning rate policy
    lr_scheduler_step_size = epochs // 2
    lr_scheduler_gamma = 0.1

    # How many iterations to print the training result
    print_frequency = 200

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"./results/test/{exp_name}"
    hr_dir = f"./data/Set5/GTmod12"

    model_path = "/user/home/al18709/work/SRResNet/SRResNet_x4-ImageNet-2096ee7f.pth.tar"
