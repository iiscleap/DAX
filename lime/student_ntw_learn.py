import os
import logging
from random import shuffle
from turtle import forward
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torchvision import models, transforms
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from skimage.color import rgb2gray
from sklearn import preprocessing
from pdb import set_trace as bp

#Fix the seeds for deterministic behavior

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# min max scaler for use
min_max_scaler = preprocessing.MinMaxScaler()

#student network uses downscaled images for highre receptive field
def get_student_transform(): 
    transf = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.ColorJitter(hue=.2, saturation = 0.2, contrast = 0.2),
        transforms.Resize((128, 128)),
        transforms.CenterCrop(128),
        # transforms.ToTensor()
    ]) 
    return transf

#incorporating color diversity is important to overcome color sensitivity
#RGB2HSV
def rgb2hsv(input, epsilon=1e-10):
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    if h.shape[0]==0: pass
    else: h, s, v = h/h.max(), s/s.max(), v/v.max()

    return torch.stack((h, s, v), dim=1)

#HSV2RGB
def hsv2rgb(input):
    assert(input.shape[1] == 3)

    h, s, v = input[:, 0], input[:, 1], input[:, 2]
    h_ = (h - torch.floor(h / 360) * 360) / 60
    c = s * v
    x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

    zero = torch.zeros_like(c)
    y = torch.stack((
        torch.stack((c, x, zero), dim=1),
        torch.stack((x, c, zero), dim=1),
        torch.stack((zero, c, x), dim=1),
        torch.stack((zero, x, c), dim=1),
        torch.stack((x, zero, c), dim=1),
        torch.stack((c, zero, x), dim=1),
    ), dim=0)

    index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
    rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0)
    return rgb

#RGB2BGR
def rgb2bgr(input):
    assert len(input.shape)==4
    input[:,[0,1,2], :, :] = input[:,[2,1,0], :, :]
    return input

# #DAME mask learning network
# class MaskLearningNetwork(nn.Module):
#     def __init__(self):
#         super(MaskLearningNetwork, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 32)
#         self.conv2 = self._conv_layer_set(32, 1)
#         self.conv3 = self._conv_layer_set(3, 32)
#         self.conv4 = self._conv_layer_set(32, 1)
#         self.conv5 = self._conv_layer_set(3, 32)
#         self.conv6 = self._conv_layer_set(32, 1)
#         #self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(8, stride=8)
#         self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
#         # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')
#         #self.W2 = nn.Linear(units)
#         #self.V = nn.Linear(units, 1)
    
#     def _conv_layer_set(self, in_c, out_c):
#         n_kernel = (5,5)
#         n_pad = int(n_kernel[0]/2)
#         conv_layer = nn.Sequential(
#         nn.Conv2d(in_c, out_c, kernel_size=n_kernel, padding=n_pad),
#         nn.LeakyReLU(),
#         )
#         return conv_layer

#     def forward(self, ip_images):
#         #bp()
#         #out_mask = self.maxpool(ip_images)
#         out_mask1 = self.conv2(self.conv1(ip_images))
#         out_mask1 = self.relu(out_mask1)
#         out_mask2 = self.conv4(self.conv3(ip_images))
#         out_mask2 = self.relu(out_mask2)
#         out_mask3 = self.conv6(self.conv5(ip_images))
#         out_mask3 = self.relu(out_mask3)
#         out_mask = torch.cat((out_mask1, out_mask2, out_mask3), dim=1)
#         #out_mask = self.conv1(out_mask)
#         #out_mask = self.conv2(out_mask)
#         #return self.sigmoid(out_mask)
#         #out_mask = self.relu(out_mask)
#         #out_mask = out_mask/torch.max(out_mask)
#         out_mask = self.maxpool(out_mask)
#         out_mask = self.upsample(out_mask)
#         #out_mask = torch.stack([self.relu((i-torch.mean(i))/(torch.max(i)-torch.mean(i))) for i in out_mask])
#         out_mask = torch.stack([self.relu(i/torch.max(i)) for i in out_mask])
#         # out_mask[out_mask>0] = 1
#         return out_mask

#DAME mask learning network
class MaskLearningNetwork(nn.Module):
    def __init__(self):
        super(MaskLearningNetwork, self).__init__()
        # self.conv1 = self._conv_layer_set(3, 32)
        # self.conv2 = self._conv_layer_set(32, 1)
        # self.conv3 = self._conv_layer_set(3, 32)
        # self.conv4 = self._conv_layer_set(32, 1)
        # self.conv5 = self._conv_layer_set(3, 32)
        # self.conv6 = self._conv_layer_set(32, 1)

        self.conv1 = self._conv_layer_set(3, 32)
        self.conv2 = self._conv_layer_set(32, 1)
        # self.conv3 = self._conv_layer_set(16, 1)
        # self.conv2 = self._conv_layer_set(8, 8)
        # self.conv3 = self._conv_layer_set(8, 3)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(8, stride=8)
        self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
        # self.upconv1 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)(scale_factor=8, mode='nearest')
        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')
        #self.W2 = nn.Linear(units)
        #self.V = nn.Linear(units, 1)
    
    def _conv_layer_set(self, in_c, out_c):
        n_kernel = (7,7)
        n_pad = int(n_kernel[0]/2)
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=n_kernel, padding=n_pad),
        nn.LeakyReLU(),
        )
        return conv_layer
    def forward(self, ip_images):
        #bp()
        # #out_mask = self.maxpool(ip_images)
        # out_mask1 = self.conv2(self.conv1(ip_images))
        # out_mask1 = self.relu(out_mask1)
        # out_mask2 = self.conv4(self.conv3(ip_images))
        # out_mask2 = self.relu(out_mask2)
        # out_mask3 = self.conv6(self.conv5(ip_images))
        # out_mask3 = self.relu(out_mask3)
        # out_mask = torch.cat((out_mask1, out_mask2, out_mask3), dim=1)

        out_mask = self.conv1(ip_images)
        out_mask = self.conv2(out_mask)
        # out_mask = self.conv3(out_mask)
        # out_mask = self.conv3(out_mask)

        #out_mask = self.conv1(out_mask)
        #out_mask = self.conv2(out_mask)
        #return self.sigmoid(out_mask)
        #out_mask = self.relu(out_mask)
        #out_mask = out_mask/torch.max(out_mask)
        out_mask = self.maxpool(out_mask)
        out_mask = self.upsample(out_mask)
        #out_mask = torch.stack([self.relu((i-torch.mean(i))/(torch.max(i)-torch.mean(i))) for i in out_mask])
        
        # out_mask = torch.stack([self.relu(i/torch.max(i)) for i in out_mask])
        # out_mask = torch.stack([self.relu(i)/torch.max(self.relu(i)) for i in out_mask])
        # out_mask = self.relu(out_mask)
        out_mask = self.leakyrelu(out_mask)
        out_mask = (out_mask - out_mask.min())/(out_mask.max() - out_mask.min())
        # out_mask = self.sigmoid(out_mask)
        # out_mask = torch.stack([(i - i.min())/(i.max() - i.min()) for i in out_mask])
        out_mask = torch.concat([out_mask, out_mask, out_mask], dim=1)
        
        # out_mask[out_mask>0] = 1
        return out_mask

# #DAME student regression network
# class StudentClassifier(nn.Module):
#     def __init__(self):
#         super(StudentClassifier, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 32)
#         self.conv2 = self._conv_layer_set(32, 8)
#         # self.fc1 = nn.Linear(25088, 128)
#         self.fc1 = nn.Linear(8192, 128)
#         self.fc2 = nn.Linear(128, 1)
#         self.relu = nn.LeakyReLU()
#         self.batch=nn.BatchNorm1d(128)
#         self.drop=nn.Dropout(p=0.1)
#         self.sigmoid=nn.Sigmoid()
#         #self.W2 = nn.Linear(units)
#         #self.V = nn.Linear(units, 1)
    
#     def _conv_layer_set(self, in_c, out_c):
#         n_kernel = (3,3)
#         n_pad = int(n_kernel[0]/2)
#         conv_layer = nn.Sequential(
#         nn.Conv2d(in_c, out_c, kernel_size=n_kernel, padding=n_pad),
#         nn.LeakyReLU(),
#         nn.MaxPool2d((2, 2)),
#         )
#         return conv_layer

#     def forward(self, ip_images):
#         # bp()
#         out = self.conv1(ip_images)
#         out = self.conv2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.batch(out)
#         out = self.drop(out)
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         return out

#DAME student regression network-- DNN based
class StudentClassifier(nn.Module):
    def __init__(self):
        super(StudentClassifier, self).__init__()
        # self.fc1 = nn.Linear(16384, 512)
        self.fc1 = nn.Linear(49152, 512)
        # self.fc1 = nn.Linear(8192, 128)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(512)
        self.batch2=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.1)
        self.sigmoid=nn.Sigmoid()
        #self.W2 = nn.Linear(units)
        #self.V = nn.Linear(units, 1)
    
    def _conv_layer_set(self, in_c, out_c):
        n_kernel = (3,3)
        n_pad = int(n_kernel[0]/2)
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=n_kernel, padding=n_pad),
        nn.LeakyReLU(),
        nn.MaxPool2d((2, 2)),
        )
        return conv_layer

    def forward(self, ip_images):
        out = self.fc1(ip_images.view(ip_images.size(0), -1))
        out = self.relu(out)
        out = self.batch1(out)
        out = self.drop(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.batch2(out)
        # out = self.drop(out)
        # out = self.fc3(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

#DAME student model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mask_compute = MaskLearningNetwork()
        self.student_classifier = StudentClassifier()
        #self.W2 = nn.Linear(units)
        #self.V = nn.Linear(units, 1)
        self.bins = 10
        self.min = 0
        self.max = 1
        self.sigma = 0.05
        self.delta = float(self.max - self.min) / float(self.bins)
        self.centers = float(self.min) + self.delta * (torch.arange(self.bins).float() + 0.5)
        # self.gausshist = GaussianHistogram(self.bins, self.min, self.max, self.sigma, self.delta, self.centers)
        self.linweight = nn.Linear(3, 1)

    def predict_proba(self, ip_images, ind, epoch, train):
        out_mask = self.mask_compute(ip_images)
        # out_mask = self.linweight(out_mask.view(out_mask.shape[0], out_mask.shape[2], out_mask.shape[3], -1)).view(out_mask.shape[0], -1, out_mask.shape[2], out_mask.shape[3])
        masked_img = ip_images*out_mask
        # masked_img = self.linweight(masked_img.view(masked_img.shape[0], masked_img.shape[2], masked_img.shape[3], -1)).view(masked_img.shape[0], -1, masked_img.shape[2], masked_img.shape[3])
        out_prob = self.student_classifier(masked_img)
        out_prob_comp = self.student_classifier(ip_images*(1-out_mask))
        mask_for_masking = None
        #############
        if ind==0 and train==True:
            #bp()
            #plt.imsave('results/masks/mask'+str(epoch)+'.png', out_mask[0][0].detach().cpu().numpy(), cmap='gray', vmax=1, vmin=0)
            #plt.imsave('results/masks/mask'+str(epoch)+'.png', out_mask[0].permute(1,2,0).detach().cpu().numpy())
            #plt.imsave('results/masks/mask'+str(epoch)+'.png', min_max_scaler.fit_transform(((out_mask[0][0]+out_mask[0][1]+out_mask[0][2])/3).detach().cpu().numpy()), cmap='gray', vmax=1, vmin=0)
            
            max_pool = nn.MaxPool2d(8, stride=8)
            up_sample = nn.Upsample(scale_factor=8, mode='bilinear')
            # out_mask = up_sample(max_pool(out_mask))

            # mask_tosave = ((out_mask[0][0]+out_mask[0][1]+out_mask[0][2])/3).detach().cpu().numpy()
            
            # mask_tosave_0 = up_sample(max_pool(((out_mask[0][0])).unsqueeze(dim=0).unsqueeze(dim=0))).squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            # mask_tosave_1 = up_sample(max_pool(((out_mask[0][1])).unsqueeze(dim=0).unsqueeze(dim=0))).squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            # mask_tosave_2 = up_sample(max_pool(((out_mask[0][2])).unsqueeze(dim=0).unsqueeze(dim=0))).squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            
            #.........
            #scheme 1
            # mask_tosave1, mask_tosave2, mask_tosave3 = out_mask[0][0], out_mask[0][1], out_mask[0][2]
            # weight1, weight2, weight3 = mask_tosave1[10:-10,10:-10].mean(), mask_tosave2[10:-10,10:-10].mean(), mask_tosave3[10:-10,10:-10].mean()
            # weight1, weight2, weight3 = weight1/(weight1+weight2+weight3), weight2/(weight1+weight2+weight3), weight3/(weight1+weight2+weight3)
            # mask_tosave = up_sample(max_pool((out_mask[0][0]*weight1+out_mask[0][1]*weight2+out_mask[0][2]*weight3).unsqueeze(dim=0).unsqueeze(dim=0))).squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            #scheme 2
            mask_tosave = up_sample(max_pool(((out_mask[0][0]+out_mask[0][1]+out_mask[0][2])/3).unsqueeze(dim=0).unsqueeze(dim=0))).squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            #scheme 3
            # out_mask = self.linweight(out_mask.view(out_mask.shape[0], out_mask.shape[2], out_mask.shape[3], -1)).view(out_mask.shape[0], -1, out_mask.shape[2], out_mask.shape[3])
            # mask_tosave = up_sample(max_pool((out_mask[0][0]).unsqueeze(dim=0).unsqueeze(dim=0))).squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            #scheme 4
            # mask_tosave = ((out_mask[0][0]+out_mask[0][1]+out_mask[0][2])/3).detach().cpu().numpy()
            #...........

            # mask_tosave[mask_tosave<mask_tosave.mean()] = mask_tosave.min()

            # plt.imsave('results/masks/mask'+str(epoch)+'0.png', mask_tosave_0, cmap='gray', vmax=1, vmin=0)
            # plt.imsave('results/masks/mask'+str(epoch)+'1.png', mask_tosave_1, cmap='gray', vmax=1, vmin=0)
            # plt.imsave('results/masks/mask'+str(epoch)+'2.png', mask_tosave_2, cmap='gray', vmax=1, vmin=0)

            plt.imsave('results/masks/mask'+str(epoch)+'.png', mask_tosave, cmap='gray', vmax=1, vmin=0)
            
            ip_image_for_masking = ip_images[0].permute(1,2,0).detach().cpu().numpy()
            ip_image_for_masking = (ip_image_for_masking-ip_image_for_masking.min())/(ip_image_for_masking.max()-ip_image_for_masking.min())
            mask_for_masking = (mask_tosave-mask_tosave.min())/(mask_tosave.max()-mask_tosave.min())

            #suppress border noise
            mask_for_masking[:7, :] = 0
            mask_for_masking[-7:, :] = 0
            mask_for_masking[:, :7] = 0
            mask_for_masking[:, -7:] = 0

            ip_image_for_masking[mask_for_masking<mask_for_masking.mean()+1.2*mask_for_masking.std()] = 0.6
            plt.imsave('results/masks/masked_img'+str(epoch)+'.png', ip_image_for_masking)
            # np.save('temp.npy', mask_for_masking)
            #mask_tosave = min_max_scaler.fit_transform(mask_tosave)
            # bp()
            mask_tosave = ((mask_tosave-mask_tosave.min())/(mask_tosave.max()-mask_tosave.min()))*256
            heatmap_img = cv2.applyColorMap(mask_tosave.astype(np.uint8), cv2.COLORMAP_JET)
            fin = cv2.addWeighted(heatmap_img, 1.0, ip_images[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8), 0.0, 0)
            RGBimage = cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)
            PILimage = Image.fromarray(RGBimage)
            # cv2.imwrite('results/masks/hmap'+str(epoch)+'.png', fin)
            PILimage.save('results/masks/hmap'+str(epoch)+'.png', dpi=(172,172))
            #plt.imsave('results/masks/mask'+str(epoch)+'.png', rgb2gray(out_mask[0].permute(1,2,0).detach().cpu().numpy()), cmap='gray', vmax=1, vmin=0)
        #############
        return out_prob, out_mask, mask_for_masking, out_prob_comp

    def forward(self, ip_images, targets, label_idx, weights, ind, epoch, train):
        prob, out_mask, mask_for_masking, prob_comp = self.predict_proba(ip_images, ind, epoch, train)
        
        # bp()

        # targets_other_classes = torch.cat((targets[:,:label_idx], targets[:,label_idx+1:]), dim=1)
        # prob_expand = prob.expand(-1, targets.shape[1]-1)
        
        #loss = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob, targets.view(targets.shape[0],-1)))+0.001*torch.mean(torch.abs(out_mask))
        #loss = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob, targets.view(targets.shape[0],-1)))+0.5*torch.mean(torch.abs(out_mask))+2*(torch.mean(torch.abs(out_mask[:8,:]))+torch.mean(torch.abs(out_mask[:,:8]))+torch.mean(torch.abs(out_mask[-8:,:]))+torch.mean(torch.abs(out_mask[:,-8:])))
        
        # loss = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob, targets[:,label_idx].view(targets.shape[0],-1)))+0.005*torch.mean(torch.abs(out_mask))
        # loss = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob, targets[:,label_idx].view(targets.shape[0],-1)))-10*self.kl_loss(F.log_softmax(prob_expand, dim=-1), F.softmax(targets_other_classes, dim=-1))+0.001*torch.mean(torch.abs(out_mask))
        
        loss1 = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob, targets[:,label_idx].view(targets.shape[0],-1)))
        # loss1 = torch.tensor(1)
        loss2 = torch.mean(torch.abs(out_mask))
        # loss2 = torch.tensor(1)
        # loss3 = self.kl_loss(((self.gausshist(prob.squeeze(dim=1))+0.001)/prob.shape[0]).log(), (self.gausshist(targets[:,label_idx].view(targets.shape[0],-1).squeeze(dim=1))/targets.shape[0]))
        loss3 = torch.tensor(1)
        prob_norm = prob/prob.sum()
        target_norm = targets[:,label_idx].view(targets.shape[0],-1)
        target_norm = target_norm/target_norm.sum()

        # loss4 = torch.mean(weights.view(weights.shape[0],-1)*self.kl_loss(prob.log(), targets[:,label_idx].view(targets.shape[0],-1)))
        loss4 = torch.mean(weights.view(weights.shape[0],-1)*self.kl_loss(prob_norm.log(), target_norm))
        loss5 = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob_comp, targets[:,label_idx].view(targets.shape[0],-1)))
        # loss = loss1+0.001*loss2+0.03*loss3
        # loss = loss1+0.002*loss2
        # loss = loss1+0.002*loss2+0.03*loss4
        # loss = loss1
        loss = loss1-1.0*loss5
        # loss = loss4
        # bp()
        # loss = loss1 + 5*loss4
        # loss = loss4
        # print(loss1, loss2, loss3)
        # loss = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob, targets[:,label_idx].view(targets.shape[0],-1)))
        #+0.001*self.kl_loss(((self.gausshist(prob.squeeze(dim=1))+0.001)/prob.shape[0]).log(), (self.gausshist(targets)/targets.shape[0]))
        # loss = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob, targets[:,label_idx].view(targets.shape[0],-1)))

        #loss = torch.mean(weights.view(weights.shape[0],-1)*self.loss(prob, targets.view(targets.shape[0],-1)))
        #loss = torch.mean(self.loss(prob, targets.view(targets.shape[0],-1)))
        return loss, loss1, loss2, loss3, mask_for_masking

class StudentDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs, labels, weights, device):
        'Initialization'
        self.labels = labels.to(device)
        self.inputs = inputs.to(device)
        self.weights = weights.to(device)
        # self.student_transf = get_student_transform()
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ip_x = self.inputs[index]
        # ip_x = self.student_transf(self.inputs[index])
        # ip_x = ip_x*255.0/ip_x.max()

        y = self.labels[index]
        w = self.weights[index]

        return ip_x, y, w

def kernel(d, kernel_width=0.25):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

def train(ip_images, targets, label_idx, distances, device_id, random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    #bp()
    print(f'Generating explanation for index: {label_idx}')
    weights = kernel(distances)
    # SEED = 0
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    
    img_orig = ip_images[0].detach().cpu().numpy()
    img_orig = (img_orig - img_orig.min())/(img_orig.max() - img_orig.min())
    plt.imsave('results/masks/img.png', img_orig)
    # ip_images = torch.from_numpy(np.array(ip_images)).permute((0,3,1,2)).float()
    ip_images = torch.stack(ip_images).permute(0,3,1,2)

    #targets = torch.from_numpy(targets[:, label_idx])
    targets = torch.tensor(targets)
    weights = torch.tensor(weights)
    #Split the train val data and hyperparams set
    X_train, X_val,y_train, y_val, w_train, w_val = train_test_split(ip_images, targets, weights, test_size=0.1, shuffle=False)
    BATCH_SIZE = 16
    EPOCHS = 9
    #base_lr = 1e-3
    base_lr = 1e-3

    train_dataset = StudentDataset(X_train, y_train, w_train, device_id)
    val_dataset = StudentDataset(X_val, y_val, w_val, device_id)
    #Dataloaders created
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    student_model = StudentModel().to(device)

    #Optimizer defined
    # optimizer = Adam([{'params':student_model.parameters()}], lr=base_lr)
    # optimizer = RMSprop([{'params':student_model.parameters()}], lr=base_lr, momentum=0.9)
    optimizer = SGD([{'params':student_model.parameters()}], lr=base_lr, momentum=0.9, nesterov=True)

    student_transf = get_student_transform()
    final_val_f1 = 0
    loss_epochs = []
    loss_epochs1, loss_epochs2, loss_epochs3 = [], [], []
    val_loss_epochs = []
    val_loss_epochs1, val_loss_epochs2, val_loss_epochs3 = [], [], []
    mask_expls = []
    for epoch in range(EPOCHS):
        #bp()
        tot_loss, val_loss = 0.0, 0.0
        tot_loss1, tot_loss2, tot_loss3 = 0.0, 0.0, 0.0
        val_loss1, val_loss2, val_loss3 = 0.0, 0.0, 0.0
        outputs = []
        targets = []
        #model_anc.train()
        student_model.train()
        #model_trans.train()
        for ind, (local_x, local_y, local_w) in enumerate(train_data_loader):
            # bp()
            # local_x = torch.stack([student_transf(x_) for x_ in local_x])
            local_x = student_transf(local_x)
            local_x = local_x/local_x.max()

            # if ind==0: 
            #     img_org = local_x[0].unsqueeze(0)

            ridx = torch.randperm(local_x.shape[0])
            
            if ind==0: 
                ridx = ridx[ridx!=0]
                local_x[0] = rgb2hsv(local_x[0].unsqueeze(0)).squeeze(0)
            
            # local_x[ridx[:int(local_x.shape[0]/2)], :] = rgb2hsv(local_x[ridx[:int(local_x.shape[0]/2)], :])
            
            if ind%2==0: local_x[ridx[:int(local_x.shape[0]*3/5)], :] = rgb2hsv(local_x[ridx[:int(local_x.shape[0]*3/5)], :])
            else: local_x[ridx[:int(local_x.shape[0]*2/5)], :] = rgb2hsv(local_x[ridx[:int(local_x.shape[0]*2/5)], :])
            
            # if ind%2==0: local_x[ridx[:int(local_x.shape[0]*4/5)], :] = rgb2hsv(local_x[ridx[:int(local_x.shape[0]*4/5)], :])
            # else: local_x[ridx[:int(local_x.shape[0]*1/5)], :] = rgb2hsv(local_x[ridx[:int(local_x.shape[0]*1/5)], :])
            
            # if ind==0:
            #     local_x[0] = rgb2hsv(img_org).squeeze(0)
            # local_x[ridx[:int(local_x.shape[0]/3)], :] = rgb2hsv(local_x[ridx[:int(local_x.shape[0]/3)], :])
            # local_x[ridx[int(local_x.shape[0]/3):2*int(local_x.shape[0]/3)], :] = rgb2bgr(local_x[ridx[int(local_x.shape[0]/3):2*int(local_x.shape[0]/3)], :])
            local_x = local_x*255.0
            # local_x = rgb2hsv(local_x)*255.0
            # local_x = local_x*255.0

            # bp()
            # print(ind, " out of ", str(len(train_data_loader)-1))
            #model_anc.zero_grad()
            student_model.zero_grad()
            optimizer.zero_grad()

            # loss, loss1, loss2, loss3 = student_model(local_x.to(device), local_y.to(device), label_idx, local_w.to(device), ind, epoch, train=True)
            loss, loss1, loss2, loss3, mask_for_masking = student_model(local_x, local_y, label_idx, local_w, ind, epoch, train=True)
            # if ind==0 and epoch>=3 and epoch<=8:
            if ind==0 and epoch>=0 and epoch<=5:
                assert np.array(mask_for_masking).any()!=None

                mask_expls.append(mask_for_masking)
            #pred = torch.argmax(final, dim = 1)
            #correct = compute_accuracy(final.cpu(), d['targets'])
            #bp()
            #loss = compute_loss(final.to(device), d['targets'])
            tot_loss += loss.item()
            tot_loss1 += loss1.item()
            tot_loss2 += loss2.item()
            tot_loss3 += loss3.item()
            #outputs.append(pred.cpu().int().numpy())
            #targets.append(d['targets'].cpu().int().numpy())
            #tot_correct += correct.item()
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info("Training done...Validation starting")
        #outputs = np.concatenate(outputs)
        #targets = np.concatenate(targets)
        # print(outputs)
        # print(targets)
        #train_f1 = compute_accuracy(outputs, targets)
        #model_anc.eval()
        #bp()
        student_model.eval()
        #model_trans.eval()
        with torch.no_grad():
            outputs, targets = [], []
            for ind, (local_x, local_y, local_w) in enumerate(val_data_loader):

                # local_x = torch.stack([student_transf(x_) for x_ in local_x])
                local_x = student_transf(local_x)
                local_x = local_x/local_x.max()
                ridx = torch.randperm(local_x.shape[0])[:int(local_x.shape[0]/2)]
                local_x[ridx, :] = rgb2hsv(local_x[ridx, :])
                local_x = local_x*255.0
                # local_x = rgb2hsv(local_x)*255.0
                # local_x = local_x*255.0

                # print(ind, " out of ", str(len(val_data_loader)-1))
                # loss, loss1, loss2, loss3 = student_model(local_x.to(device), local_y.to(device), label_idx, local_w.to(device),ind,epoch, train=False)
                loss, loss1, loss2, loss3, _ = student_model(local_x, local_y, label_idx, local_w,ind,epoch, train=False)

                # final = model_lstm(final)

                # pred = torch.argmax(final, dim = 1)
                #correct = compute_accuracy(final.cpu(), d['targets'])
                # loss = compute_loss(final.to(device), d['targets'])
                val_loss += loss.item()
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()
                val_loss3 += loss3.item()
                # outputs.append(pred.cpu().int().numpy())
                # targets.append(d['targets'].cpu().int().numpy())
            # outputs = np.concatenate(outputs)
            # targets = np.concatenate(targets)
            # val_f1 = compute_accuracy(outputs, targets)
            #val_correct += correct.item()
            # if val_f1 > final_val_f1:
            #         '''torch.save({'model_state_dict': model_anc.state_dict(),
            #                     'optimizer_state_dict': optimizer.state_dict(),},
            #                     'anc_model.tar')'''
            #         torch.save({'model_state_dict': model_res.state_dict(),
            #                     'optimizer_state_dict': optimizer.state_dict(),},
            #                     'res_model.tar')
            #         torch.save({'model_state_dict': model_lstm.state_dict(),
            #                     'optimizer_state_dict': optimizer.state_dict(),},
            #                     'lstm_model.tar')
            #         '''torch.save({'model_state_dict': model_trans.state_dict(),
            #                     'optimizer_state_dict': optimizer.state_dict(),},
            #                     'trans_model.tar')'''
            #         final_val_f1 = val_f1
        e_log = epoch + 1
        train_loss = tot_loss/len(train_data_loader)
        train_loss1 = tot_loss1/len(train_data_loader)
        train_loss2 = tot_loss2/len(train_data_loader)
        train_loss3 = tot_loss3/len(train_data_loader)
        #train_acc = tot_correct/df_train.shape[0]
        val_loss_log = val_loss/len(val_data_loader)
        val_loss_log1 = val_loss1/len(val_data_loader)
        val_loss_log2 = val_loss2/len(val_data_loader)
        val_loss_log3 = val_loss3/len(val_data_loader)
        #val_acc_log = val_correct/df_val.shape[0]
        loss_epochs.append(train_loss)
        loss_epochs1.append(train_loss1)
        loss_epochs2.append(train_loss2)
        loss_epochs3.append(train_loss3)
        val_loss_epochs.append(val_loss_log)
        val_loss_epochs1.append(val_loss_log1)
        val_loss_epochs2.append(val_loss_log2)
        val_loss_epochs3.append(val_loss_log3)
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log}")
    #bp()
    np.save('results/plots/train_loss.npy', loss_epochs)
    np.save('results/plots/train_loss1.npy', loss_epochs1)
    np.save('results/plots/train_loss2.npy', loss_epochs2)
    np.save('results/plots/train_loss3.npy', loss_epochs3)
    np.save('results/plots/val_loss.npy', val_loss_epochs)
    np.save('results/plots/val_loss1.npy', val_loss_epochs1)
    np.save('results/plots/val_loss2.npy', val_loss_epochs2)
    np.save('results/plots/val_loss3.npy', val_loss_epochs3)
    plt.figure()
    plt.plot(loss_epochs)
    plt.plot(val_loss_epochs)
    plt.legend(['train loss', 'val loss'])
    plt.savefig('results/plots/lossplot.png')
    plt.close()
    mask_expls = np.array(mask_expls).mean(axis=0)
    mask_expls = (mask_expls - mask_expls.min())/(mask_expls.max() - mask_expls.min())
    return mask_expls