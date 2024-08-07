from utils import *
import time
import cv2
from torchvision.transforms import ToPILImage
import torchvision


def train(train_loader, model, criterion, optimizer, epoch):
    print('Epoch: %d' % (epoch + 1))
    model.train()
    sum_loss = 0
    for inputs, _ in train_loader:
        optimizer.zero_grad()
        #inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
        inputs = inputs.cuda()
        outputs = model(inputs)
        loss = criterion(outputs[0], inputs) + criterion(outputs[1], inputs)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss


def valid_bsds(valid_loader, model, criterion):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            #inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            inputs = inputs.cuda()
            outputs = model(inputs)
            #inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255. #将图像转为YCbCr格式，只计算Y分量即亮度分量的PSNR
            output_0 = outputs[0]
            #output_0 = rgb_to_ycbcr(output_0.cuda())[:, 0, :, :].unsqueeze(1) / 255. #将图像转为YCbCr格式，只计算Y分量即亮度分量的PSNR
            mse = F.mse_loss(output_0, inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(output_0, inputs)

    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)


def valid_set(valid_loader, model, criterion):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            #inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            inputs = inputs.cuda()
            outputs = model(inputs)
            #inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            output_0 = outputs[0]
            #output_0 = rgb_to_ycbcr(output_0.cuda())[:, 0, :, :].unsqueeze(1) / 255.
            mse = F.mse_loss(output_0, inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(output_0, inputs)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)
