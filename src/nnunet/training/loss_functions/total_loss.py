from torch import nn
# import torch
import torch.nn.functional as F
from training.loss_functions.distill import DistillKL, Attention

class SelfDistillation_loss(nn.Module):
    def __init__(self, loss, weight_factors):
         super(SelfDistillation_loss, self).__init__()

         self.alpha = 0.1
         self.beta = 0.3
         self.weights = weight_factors
         self.dc_ce = loss
         self.criterion_att = Attention()
         self.criterion_kd = DistillKL()
         self.final_nonlin = lambda x: F.softmax(x, 1)


    def forward(self, seg_outputs, tea_outputs, stu_features, tea_features, targets):
        assert isinstance(seg_outputs, (tuple, list)), "seg_outputs must be either tuple or list"
        assert isinstance(targets, (tuple, list)), "targets must be either tuple or list"

        # 真实标签的损失1
        dc_ce_loss_1 = self.weights[0] * self.dc_ce(seg_outputs[0], targets[0])
        for i in range(1, len(seg_outputs)):
            if self.weights[i] != 0:
                dc_ce_loss_1 += self.weights[i] * self.dc_ce(seg_outputs[i], targets[i])

        # 真实标签的损失2
        dc_ce_loss_2 = self.weights[0] * self.dc_ce(tea_outputs[0], targets[0])
        for i in range(1, len(seg_outputs)):
            if self.weights[i] != 0:
                dc_ce_loss_2 += self.weights[i] * self.dc_ce(tea_outputs[i], targets[i])
        
        # 软标签蒸馏的损失
        # kd_loss = self.weights[0] * self.criterion_kd(stu_logits[0], tea_logits[0])
        # for i in range(1, len(seg_outputs)):
        #     if self.weights[i] != 0:
        #         kd_loss += self.weights[i] * self.criterion_kd(stu_logits[i], tea_logits[i])
        
        # 特征之间的损失
        feature_loss = self.weights[0] * self.criterion_att(stu_features[0], tea_features[0])
        for i in range(1, len(stu_features)):
            if self.weights[i] != 0:
                feature_loss += self.weights[i] * self.criterion_att(stu_features[i], tea_features[i])
        


        #total_loss = (dc_ce_loss_1 + dc_ce_loss_2)/2 + self.alpha * kd_loss
        total_loss = (dc_ce_loss_1 + dc_ce_loss_2)/2 + self.beta * feature_loss

        return total_loss


# def kd_loss_function(outputs, targets):
#     log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
#     softmax_targets = F.softmax(targets/3.0, dim=1)
#     return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

# def feature_loss_function(fea, target_fea):
#     loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
#     return torch.abs(loss).sum()