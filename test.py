import argparse
import warnings
import model.mambacs as MambaCS
from loss import *
from data_processor import *
from tester import *

warnings.filterwarnings("ignore")

import torch
from transformers import BertModel
import numpy as np

import torch
from thop import profile
import datetime

def main():
    global args
    args = parser.parse_args()
    print(args)

    if args.model == 'mambacs':
        model = MambaCS.MambaCS(sensing_rate=args.sensing_rate, im_size=256)

    model = model.cuda()

    dic = './best/' + str(args.model) + '_' + str(args.sensing_rate) + '.pth'

    model.load_state_dict(torch.load(dic)['state_dict'], strict=False)
    criterion = loss_fn

    trn_loader, bsds, set5, set14 = data_loader(args)
    model.eval()

    psnr1, ssim1 = valid_BSDS(bsds, model, criterion, args)
    print("----------BSDS----------PSNR: %.2f----------SSIM: %.4f" % (psnr1, ssim1))
    psnr2, ssim2 = valid_set5(set5, model, criterion, args)
    print("----------Set5----------PSNR: %.2f----------SSIM: %.4f" % (psnr2, ssim2))
    psnr3, ssim3 = valid_set14(set14, model, criterion, args)
    print("----------Set14----------PSNR: %.2f----------SSIM: %.4f" % (psnr3, ssim3))


if __name__ == '__main__':
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mambacs',
                        help='choose model to train')
    parser.add_argument('--sensing-rate', type=float, default=0.50000,
                        help='set sensing rate')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--block-size', default=32, type=int,
                        metavar='N', help='block size (default: 32)')
    parser.add_argument('--image-size', default=96, type=int,
                        metavar='N', help='image size used for training (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)

    main()
