#!/usr/bin/env bash
python train.py --dataroot ./datasets/block/foldAB/ --name blocks_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --input_nc 6 --dataset_mode multi --no_flip
