import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class CrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0, beta=0, gamma=0, size_average=True, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index,size_average=self.size_average)
        if self.cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, logit, target, features_in):
        n, c, h, w = logit.size()
        
        CE_loss = self.criterion(logit, target.long())
        return CE_loss / n
        VAR_loss = Variable(torch.Tensor([0])).cuda()
        Inter_loss = Variable(torch.Tensor([0])).cuda()
        Center_loss = Variable(torch.Tensor([0])).cuda()
        for i in range(n):
            label = target[i]
            label = label.flatten().cpu().numpy()
            features = logit[i]
            features = features.permute(1, 2, 0).contiguous()
            shape = features.size()
            features = features.view(shape[0]*shape[1], shape[2])
            features_in_temp = features_in[i]

            instances, counts = np.unique(label, False, False, True)
            # print('counts', counts)
            total_size = int(np.sum(counts))
            for instance in instances:

                if instance == self.ignore_index:  # Ignore background
                    continue

                locations = torch.LongTensor(np.where(label == instance)[0]).cuda()
                vectors = torch.index_select(features, dim=0, index=locations)
                features_temp = torch.index_select(features_in_temp, dim=0, index=locations)
                centers_temp = torch.mean(features_temp, dim=0)
                features_temp = features_temp - centers_temp
                Center_loss += torch.sum(features_temp ** 2) / total_size
                # print(size)
                # print(-vectors[:,int(instance)])
                # get instance mean and distances to mean of all points in an instance
                VAR_loss += torch.sum((-vectors[:,int(instance)]))/total_size
                Inter_loss += (torch.sum(vectors) - torch.sum((vectors[:,int(instance)]))) / total_size

                # total_size += size

            # VAR_loss += var_loss/total_size

        loss = (CE_loss + self.alpha * VAR_loss + self.beta * Inter_loss +self.gamma * Center_loss) / n
        # print(CE_loss/n, self.alpha * VAR_loss/n, self.beta * Inter_loss/n, self.gamma * Center_loss/n)

        return loss

class CrossEntropyLoss_dis(nn.Module):
    def __init__(self, alpha=0, beta=0, gamma=0, size_average=True, ignore_index=255):
        super(CrossEntropyLoss_dis, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, logit, target, features_1, features_2):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index,size_average=self.size_average)

        if self.cuda:
            criterion = criterion.cuda()

        CE_loss = criterion(logit, target.long())

        return CE_loss / n

        DIS_loss = Variable(torch.Tensor([0])).cuda()

        appendix_lay = torch.zeros(n,w,h,1).cuda()
        features_1 = torch.cat((features_1, appendix_lay), dim=3)
        # print('features_1.shape: ', features_1.shape)
        # print('features_2.shape: ', features_2.shape)

        for i in range(n):
            features_origin = features_1[i][target[i] != 16]
            features_new = features_2[i][target[i] != 16]
            features_diff = features_new - features_origin
            DIS_loss += torch.sum(features_diff ** 2) / (features_diff.shape[0])

        loss = CE_loss / n + 0.01 * DIS_loss / n
        # print(CE_loss, DIS_loss)



        return loss

# class CenterLoss(nn.Module):
#     """Center loss.
    
#     Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#     """
#     def __init__(self, num_classes=10, feat_dim=256, use_gpu=True):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu

#         if self.use_gpu:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda()) # (C, M)
#         else:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim, h, w).
#             labels: ground truth labels with shape (batch_size, h, w).
#         """
#         batch_size = x.size(0)
#         x = x.permute(0, 2, 3, 1) # (B, H, W, M)
    
#         x = x.reshape((-1,self.feat_dim)) # (N, M)
#         sample_size= x.size(0) # N
#         labels = labels.flatten() # (N,)
#         assert sample_size == labels.size(0)
#         # (N, M) --> (N, 1) --> (N, C) | (C, M) --> (C, 1) --> (C, N) --> (N, C)
#         # (N, C)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(sample_size, self.num_classes) + \
#                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, sample_size).t()
#         # distmat - 2 (x * center.T)
#         distmat.addmm_(1, -2, x, self.centers.t())

#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(sample_size, self.num_classes)
#         mask = labels.eq(classes.expand(sample_size, self.num_classes))

#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / sample_size

#         return loss / batch_size

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda()) # (C, M)
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            self.criterion = nn.CrossEntropyLoss()

    def _dis_criterion(self, x, labels):
        # x: (B, M, H, W) | labels: (B, H, W)
        _, _, H, W = x.shape 
        assert H == W
        x = torch.nn.functional.interpolate(x, size=[H//2, W//2])
        labels = torch.nn.functional.interpolate(labels.unsqueeze(dim=1).float(), size=[H//2, W//2], mode="nearest")
        logit = [-torch.sum((x.unsqueeze(dim=1) - self.centers.clone()[c:c+1, :].detach().view(1, 1, self.centers.shape[1], 1, 1)) ** 2, dim=2) for c in range(self.num_classes)]
        logit = torch.cat(logit, dim=1)
        logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        label = labels.contiguous().view(-1)
        #logit = -torch.sum((x.unsqueeze(dim=1) - self.centers.clone().detach().view(1, *self.centers.shape, 1, 1)) ** 2, dim=2)
        loss = self.criterion(logit[label != 255], label[label != 255].long())
        return loss 

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim, h, w).
            labels: ground truth labels with shape (batch_size, h, w).
        """
        # feature = x.clone()
        # feature_label = labels.clone()

        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1) # (B, H, W, M)
    
        x = x.reshape((-1,self.feat_dim)) # (N, M)
        sample_size= x.size(0) # N
        labels = labels.flatten() # (N,)
        assert sample_size == labels.size(0)
        # (N, M) --> (N, 1) --> (N, C) | (C, M) --> (C, 1) --> (C, N) --> (N, C)
        # (N, C)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(sample_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, sample_size).t()
        # distmat - 2 (x * center.T)
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(sample_size, self.num_classes)
        mask = labels.eq(classes.expand(sample_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / sample_size

        #norm_loss = torch.exp(-torch.norm(self.centers.unsqueeze(dim=0)-self.centers.unsqueeze(dim=1), p=2, dim=-1))

        #dis_loss = self._dis_criterion(feature, feature_label)

        return loss / batch_size #+ norm_loss / batch_size

if __name__ =='__main__':
    center_loss=CenterLoss()
    print(center_loss.centers.data.shape)
    center=center_loss.centers.data
    torch.save(center,'center.pth')
    #torch.save('./center.pth',center_loss.state_dict())