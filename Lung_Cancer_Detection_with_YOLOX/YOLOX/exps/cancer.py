#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
from yolox.data.dataloading import get_yolox_datadir

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/cancer"
        self.train_ann = "train_data.json"
        self.val_ann = "test_data.json"

        self.num_classes = 4

        self.max_epoch = 50
        self.data_num_workers = 4
        self.eval_interval = 1

    # def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
    #     from yolox.data import (
    #         COCODataset,
    #         TrainTransform,
    #         YoloBatchSampler,
    #         DataLoader,
    #         InfiniteSampler,
    #         MosaicDetection,
    #         worker_init_reset_seed,
    #     )
    #     from yolox.utils import (
    #         wait_for_the_master,
    #         get_local_rank
    #     )

    #     local_rank = get_local_rank()

    #     with wait_for_the_master(local_rank):
    #         dataset = COCODataset(

    #         )
    #     return super().get_data_loader(batch_size, is_distributed, no_aug=no_aug, cache_img=cache_img)