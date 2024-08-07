from utils import *
import time
import cv2
from torchvision.transforms import ToPILImage
import torchvision
from skimage import util
import matplotlib.pyplot as plt

def show_image1(output_tensor,iters,args):
        # 将输出的张量转换为 numpy 数组
        output_numpy = output_tensor.cpu().squeeze()
        # 将通道维度放在最后，适用于 matplotlib
        output_numpy = ToPILImage()(output_numpy)
        # 显示图像
        output_numpy.save(str(args.sensing_rate)+'/set5_'+str(iters)+'.png')

def show_image2(output_tensor,iters,args):
        # 将输出的张量转换为 numpy 数组
        output_numpy = output_tensor.cpu().squeeze()
        # 将通道维度放在最后，适用于 matplotlib
        output_numpy = ToPILImage()(output_numpy)
        #output_numpy = np.transpose(output_numpy, (1, 2, 0))
        # 显示图
        output_numpy.save(str(args.sensing_rate)+'/set14_'+str(iters)+'.png')

def show_image3(output_tensor,iters,args, psnr, ssim):
        # 将输出的张量转换为 numpy 数组
        output_numpy = output_tensor.cpu().squeeze()
        output_numpy = ToPILImage()(output_numpy)
        #output_numpy = np.transpose(output_numpy, (1, 2, 0))
        # 显示图像
        output_numpy.save(str(args.sensing_rate)+'/BSDS_'+str(iters)+'-'+str(psnr)+'-'+str(ssim)+'.png')


def valid_set5(valid_loader, model, criterion, args):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            #inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            output_0 = outputs[0]
            #output_0 = rgb_to_ycbcr(output_0.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            mse = F.mse_loss(output_0, inputs)
            psnr = 10 * log10(1 / mse.item())
            #print("psnr:%.2f" % (psnr))
            ssim0 = ssim(output_0, inputs)
            #print("ssim:%.4f" % (ssim0))
            sum_psnr += psnr
            sum_ssim += ssim0
            #show_image1(output_0, iters, args)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)

def valid_set14(valid_loader, model, criterion,args):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            #inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            output_0 = outputs[0]
            #output_0 = rgb_to_ycbcr(output_0.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            mse = F.mse_loss(output_0, inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(output_0, inputs)
            #show_image2(output_0, iters, args)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)

def valid_BSDS(valid_loader, model, criterion,args):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            #inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            output_0 = outputs[0]
            #output_0 = rgb_to_ycbcr(output_0.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            mse = F.mse_loss(output_0, noisy1)
            psnr = 10 * log10(1 / mse.item())
            ssim0 = ssim(output_0, inputs)
            sum_psnr += psnr
            sum_ssim += ssim0
            #show_image3(output_0, iters, args, psnr, ssim0)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)