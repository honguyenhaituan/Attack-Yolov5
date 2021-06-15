import torch
from utils.loss import ComputeLoss
from utils.torch_utils import time_synchronized
from torch.autograd.gradcheck import zero_gradients
from attack.utils_image import blur_image
from attack.metrics import psnr, cosine

class I_FGSM: 
    def __init__(self, max_iter=100, epsilon=10/255): 
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.alpha = self.epsilon / self.max_iter
        self.grad = None
        self.iter = 0

    def refresh(self):
        self.iter = 0
        self.grad = None

    def get_max_iter(self): 
        return self.max_iter

    def _update_grad(self, grad): 
        self.grad = grad

    def get_update_image(self, grad):
        self.iter += 1
        self._update_grad(grad)
        return self.alpha * torch.sign(self.grad)

class MI_FGSM(I_FGSM):
    def __init__(self, max_iter=100, epsilon=10/255, momentum=0.9): 
        super(MI_FGSM, self).__init__(max_iter, epsilon)
        self.momentum = momentum

    def _update_grad(self, grad): 
        if self.grad == None or self.grad.shape != grad.shape: 
            self.grad = torch.zeros_like(grad)

        self.grad = self.momentum * self.grad + grad / torch.sum(torch.abs(grad))

class DMI_FGSM(MI_FGSM):
    def __init__(self, max_iter=100, epsilon=10/255, momentum=0.9, decay=0.95):
        super(DMI_FGSM, self).__init__(max_iter, epsilon, momentum)
        self.decay = decay
        if decay != 1: 
            self.factor = self.epsilon * (1 - decay) / (1 - decay ** max_iter)
        else: 
            self.factor = self.epsilon / max_iter

    def get_update_image(self, grad):
        self.momentum = self.factor * (self.decay ** self.iter)
        return super().get_update_image(grad)

class D2MI_FGSM(MI_FGSM):
    def __init__(self, max_iter=100, epsilon=10/255, momentum=0.9, decay=0.1):
        super(D2MI_FGSM, self).__init__(max_iter, epsilon, momentum)
        self.decay = decay
        sum = 0
        for i in range(max_iter):
            sum += 1 / (1 + decay * i)
        self.factor = epsilon / sum
        self.alpha = self.factor / (1 + decay * self.iter)
        
    def get_update_image(self, grad):
        self.alpha = self.factor / (1 + self.decay * self.iter)
        return super().get_update_image(grad)

class D3MI_FGSM(MI_FGSM):
    def __init__(self, max_iter=100, epsilon=10/255, momentum=0.9):
        super(D3MI_FGSM, self).__init__(max_iter, epsilon, momentum)
        self.momentum0 = momentum

    def get_update_image(self, grad):
        p_i = (self.max_iter - self.iter) / self.max_iter
        self.momentum = self.momentum0 * (p_i / (1 - self.momentum0 + self.momentum0 * p_i))
        return super().get_update_image(grad)

def get_method_attack(name_attack, max_iter, epsilon, momentum, decay):
    if name_attack == 'I-FGSM': 
        return I_FGSM(max_iter, epsilon)
    if name_attack == 'MI-FGSM':
        return MI_FGSM(max_iter, epsilon, momentum)

    return None

@torch.no_grad()
def attack_images(model, img, targets, method_attack, logger, no_blur=False):
    compute_loss = ComputeLoss(model)

    t = time_synchronized()
    if no_blur:
        att_img = img.clone()
        bs, _, h, w = att_img.shape
        masks = torch.zeros((bs, h, w), dtype=torch.bool)
    else: 
        att_img, masks = blur_image(img, targets)

    att_img.requires_grad = True
    time_process = time_synchronized() - t
    
    if logger:
        logger.increase_log({"time/process_attack":time_process})

    method_attack.refresh()
    for i in range(method_attack.get_max_iter()):
        t = time_synchronized()

        zero_gradients(att_img)
        with torch.set_grad_enabled(True):
            _, train_out = model(att_img)
            loss, _ = compute_loss(train_out, targets)
        loss.backward()

        updates_image = method_attack.get_update_image(att_img.grad)
        for update_image, mask in zip(updates_image, masks):
            update_image[:, mask] = 0

        att_img.data = torch.clamp(att_img - updates_image, 0, 1)

        time_attack = time_synchronized() - t
        if logger:
            logger.increase_log({"time/iterate_attack":time_attack})
            logger.increase_log_epoch(i, {'loss': loss.item()})
            logger.increase_log_epoch(i, {'metrics/cosine': cosine(img, att_img)})
            logger.increase_log_epoch(i, {'metrics/psnr': psnr(img, att_img)})

    return att_img

# @torch.enable_grad()
# def attack_images(model, img, targets, method_attack):
#     compute_loss = ComputeLoss(model)

#     att_img, masks = blur_image(img, targets)
#     bs, c, h, w = img.shape
#     delta = torch.ones((bs, h, w))
#     # delta = torch.ones_like(att_img)
#     delta.requires_grad = True

#     optimizer = torch.optim.SGD([delta], lr=3, momentumum=0.9)

#     for i in range(method_attack.get_max_iter()):
#         optimizer.zero_grad()
#         #TODO: the inference main model have argument model(img, augment=augment) 
#         _att_img = torch.like(att_img)
        
#         _, train_out = model(att_img * delta)

#         #calculate loss like train.py
#         loss, _ = compute_loss(train_out, targets)
#         print(i, loss)
#         loss.backward()

#         # updates_image = method_attack.get_update_image(delta.grad)
#         # for update_image, mask in zip(updates_image, masks):
#         #     update_image[:, mask] = 0
        
#         delta.grad.data = transforms.GaussianBlur(9)(delta.grad.data)
#         for _grad, mask in zip(delta.grad, masks):
#             _grad[:, mask] = 0.0

#         optimizer.step()
        
#         delta.data = torch.where(att_img * delta > 1, 1 / att_img, delta)
#         delta.data = torch.clamp(delta, min=0.0)

#     print(loss)
#     return att_img * delta