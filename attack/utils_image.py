import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import ImageFilter

from utils.general import xywh2xyxy

def blur_image(images, targets, radius=10):
    blur_imgs = []
    masks = []
    _, _, h, w = images.shape

    for si, image in enumerate(images): 
        target = targets[targets[:, 0] == si]
        boxes = xywh2xyxy(target[:, 2:6])
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

        img = transforms.ToPILImage()(image)
        blur_img = img.filter(ImageFilter.GaussianBlur(radius))

        mask = np.ones(image.shape[-2:], dtype=bool)
        for box in boxes: 
            box = box.to('cpu').tolist()
            box = [round(x) for x in box]
            mask[box[1]:box[3], box[0]:box[2]] = 0

            crop_img = blur_img.crop(box)
            img.paste(crop_img, box)

        blur_imgs.append(transforms.ToTensor()(img))
        masks.append(torch.from_numpy(mask))

    blur_imgs = torch.stack(blur_imgs).to(images.device)
    masks = torch.stack(masks).to(images.device)
    if images.dtype == torch.half or images.dtype == torch.float16: 
        blur_imgs = blur_imgs.half()

    return blur_imgs, masks

def cvtOriginalImage(image, shape): 
    ori_shape, (ratio, (dw, dh)) = shape
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    org_image = image[:, top : -bottom, left : -right]
    org_image = F.interpolate(org_image.unsqueeze(dim=0), ori_shape)

    return org_image.squeeze(dim=0)

def save_images(images, paths, shapes):
    for image, path, shape in zip(images, paths, shapes): 
        ori_shape, (ratio, (dw, dh)) = shape
        org_image = transforms.ToPILImage()(image)
        org_image = cv2.cvtColor(np.asarray(org_image),cv2.COLOR_RGB2BGR)      
        # print(org_image.shape, ori_shape, ratio, dw, dh)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        org_image = org_image[top : -bottom, left : -right, :]
        org_image = cv2.resize(org_image, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(str(path), org_image)
