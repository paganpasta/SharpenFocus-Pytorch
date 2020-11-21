import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial
import models.resnet as resnet

class SFOCUS(nn.Module):
    def __init__(self, model, grad_layers, num_classes):
        super(SFOCUS, self).__init__()
        self.model = model
        # print(self.model)
        self.grad_layers = grad_layers

        self.num_classes = num_classes

        # Feed-forward features
        self.feed_forward_features = {}
        # Backward features
        self.backward_features = {}

        # Register hooks
        self._register_hooks(grad_layers)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100

    def _register_hooks(self, grad_layers):
        def forward_hook(name, module, grad_input, grad_output):
            self.feed_forward_features[name] = grad_output

        def backward_hook(name, module, grad_input, grad_output):
            self.backward_features[name] = grad_output[0]

        gradient_layers_found = 0
        for idx, m in self.model.named_modules():
            if idx in self.grad_layers:
                m.register_forward_hook(partial(forward_hook, idx))
                m.register_backward_hook(partial(backward_hook, idx))
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layers_found += 1
                

        # for our own sanity, confirm its existence
        if gradient_layers_found != 2:
            raise AttributeError('Gradient layers %s not found in the internal model' % grad_layers)

    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_classes), requires_grad=False).cuda()
        ohe.scatter_(1, labels.unsqueeze(1), 1)
        return ohe

    def populate_grads(self, logits, labels_ohe):
        gradient = logits * labels_ohe
        grad_logits = (logits * labels_ohe).sum()  # BS x num_classes
        grad_logits.backward(gradient=grad_logits, retain_graph=True)
        self.model.zero_grad()
    
    def loss_attention_separation(self, At, Aconf):
        At_min = At.min().detach()
        At_max = At.max().detach()
        scaled_At = (At - At_min)/(At_max - At_min)
        sigma = 0.25 * At_max
        omega = 100.
        mask = F.sigmoid(omega*(scaled_At-sigma))
        L_as_num = (torch.min(At, Aconf)*mask).sum() 
        L_as_den = (At+Aconf).sum()
        L_as = 2.0*L_as_num/L_as_den

        return L_as, mask
    
    def loss_attention_consistency(self, At, mask):
        theta = 0.8
        num = (At*mask).sum()
        den = At.sum()
        L_ac = theta - num/den
        return L_ac

    def forward(self, images, labels):

        #For testing call the function in eval mode and remove with torch.no_grad() if any

        logits = self.model(images)  # BS x num_classes
        self.model.zero_grad()

        _, indices = torch.topk(logits, 2)
        preds = indices[:, 0]
        seconds = indices[:, 1]
        good_pred_locs = torch.where(preds.eq(labels)==True)
        preds[good_pred_locs] = seconds[good_pred_locs]

        #Now preds only contains indices for confused non-gt classes
        conf_1he = self._to_ohe(preds).cuda()
        gt_1he = self._to_ohe(labels).cuda()
        
        #Store attention w.r.t correct labels
        
        self.populate_grads(logits, conf_1he)
        #Store attention w.r.t confused labels
        for idx, name in enumerate(self.grad_layers):
            if idx == 0:
                 backward_feature = self.backward_features[name]
                 forward_feature  = self.feed_forward_features[name]
                 weights = F.adaptive_avg_pool2d(F.relu(backward_feature), 1)
                 A_conf_in = F.relu(torch.mul(forward_feature, weights).sum(dim=1, keepdim=True))
            else:
                 backward_feature = self.backward_features[name]
                 forward_feature = self.feed_forward_features[name]
                 weights = F.adaptive_avg_pool2d(F.relu(backward_feature), 1)
                 A_conf_la = F.relu(torch.mul(forward_feature, weights).sum(dim=1, keepdim=True))
        
        self.populate_grads(logits, gt_1he)
        for idx, name in enumerate(self.grad_layers):
            if idx == 0:
                 backward_feature = self.backward_features[name]
                 forward_feature  = self.feed_forward_features[name]
                 weights = F.adaptive_avg_pool2d(F.relu(backward_feature), 1)
                 A_t_in = F.relu(torch.mul(forward_feature, weights).sum(dim=1, keepdim=True))
            else:
                 backward_feature = self.backward_features[name]
                 forward_feature = self.feed_forward_features[name]
                 weights = F.adaptive_avg_pool2d(F.relu(backward_feature), 1)
                 A_t_la = F.relu(torch.mul(forward_feature, weights).sum(dim=1, keepdim=True))
        
        #Loss Attention Separation 
        L_as_la, _ = self.loss_attention_separation(A_t_la, A_conf_la)
        L_as_in, mask_in = self.loss_attention_separation(A_t_in, A_conf_in)
        #Loss Attention Consistency
        L_ac_in = self.loss_attention_consistency(A_t_in, mask_in)
        return logits, L_as_la, L_as_in, L_ac_in, A_t_la
        
      
def sfocus18(num_classes, pretrained=False):
    grad_layers = ['layer3', 'layer4']
    base = resnet.resnet18(num_classes=num_classes)
    model = SFOCUS(base, grad_layers, num_classes)
    return model


if __name__ == '__main__':
    model = sfocus18('large', 10).cuda()
    sample_x = torch.randn([5, 3, 224, 224])
    sample_y = torch.tensor([i for i in range(5)])
    model.train()
    a, b, c, d, e = model(sample_x.cuda(), sample_y.cuda())
    print(a, b, c, d, e.shape)
