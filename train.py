import argparse
import os
import warnings
import shutil
import model.mambacs as mambacs
from loss import *
import torch.optim as optim
from data_processor import *
from trainer import *

warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()
    # setup_seed(1)

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        torch.backends.cudnn.benchmark = True

    if args.model == 'mambacs':
        model = mambacs.MambaCS(sensing_rate=args.sensing_rate, im_size=args.image_size)

    model = model.cuda()
    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90, 120, 150, 180], gamma=0.25, last_epoch=-1)
    train_loader, test_loader_bsds, test_loader_set5, test_loader_set14 = data_loader(args)

    print('\nModel: %s\n'
          'Sensing Rate: %.6f\n'
          'Epoch: %d\n'
          'Initial LR: %f\n'
          % (args.model, args.sensing_rate, args.epochs, args.lr))

    print('Start training')
    psnr = 0
    ssim = 0
    for epoch in range(args.epochs):
        print('\ncurrent lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        psnr1, ssim1 = valid_bsds(test_loader_bsds, model, criterion)
        print("----------BSDS----------PSNR: %.2f----------SSIM: %.4f" % (psnr1, ssim1))
        psnr2, ssim2 = valid_set(test_loader_set5, model, criterion)
        print("----------Set5----------PSNR: %.2f----------SSIM: %.4f" % (psnr2, ssim2))
        psnr3, ssim3 = valid_set(test_loader_set14, model, criterion)
        print("----------Set14----------PSNR: %.2f----------SSIM: %.4f" % (psnr3, ssim3))
        if (epoch % 5 == 0):
            psnr0 = psnr
            ssim0 = ssim
            psnr = psnr1 + psnr2 + psnr3
            ssim = ssim1 + ssim2 + ssim3       
            is_best = (psnr > psnr0) or (ssim > ssim0)
            save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.model,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best, args.save_dir + '/ckp_CS_' + str(epoch) + 'checkpoint.pth')

    print('Trained finished.')

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mambacs',
                        help='choose model to train')
    parser.add_argument('--sensing-rate', type=float, default=0.5,
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
                        help='The directory used to save the trained model',
                        default='save_temp', type=str)

    main()
