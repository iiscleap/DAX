import argparse
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# import torchvision.datasets as datasets
# import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from tqdm import tqdm
import pickle

from xai_VOC import *
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--exp_class', type=int, required=True)
args = parser.parse_args()

dataset_dir = '/home/debarpanb/debarpanb/XAI_exp/VOC_bbox_refined_dataset/resnet/VOC_filtered/correct_preds_and_annotations.pt'
model_path = '/home/debarpanb/debarpanb/XAI_exp/finetune_bbox_VOC/resnet/models/resnet18-finetuned-voc.pt'
save_explanations_path = 'explanation_masks'

# voc_cls_maps = np.load('utilities/voc_cls.npy')

class VOC_XAI_Dataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, images, labels, annts, device):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images.to(device)
        self.labels = labels.to(device)
        self.annts = annts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.images[idx], self.labels[idx], self.annts[idx]

device_id = "cuda:0"
device = torch.device(device_id if torch.cuda.is_available() else "cpu")

images, labels, annts = torch.load(dataset_dir)
idx_cls = labels==args.exp_class
images, labels, annts = images[idx_cls], labels[idx_cls], annts[idx_cls.detach().cpu().numpy()]

xai_dataset = VOC_XAI_Dataset(torch.tensor(images), torch.tensor(labels), annts, device)
xai_dataset_size = len(xai_dataset)
xai_dataloader = torch.utils.data.DataLoader(dataset=xai_dataset,
                                    batch_size=1,
                                    shuffle=False)

model = models.resnet18(pretrained=True)

model.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 21)

model.load_state_dict(torch.load(model_path, map_location=device_id))
model = model.to(device)
 
def get_preprocess_transform():
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])     
    # transf = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize
    # ])   

    transf_fn = torch.nn.Sequential(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    transf = torch.jit.script(transf_fn) 

    return transf 

preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    # batch = torch.stack(tuple(localtransf(i) for i in images), dim=0)
    # batch = images.permute(0,3,1,2)

    images = images/images.max()
    batch = preprocess_transform(images.permute(0,3,1,2))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device(device_id)
    # model.to(device)
    # batch = batch.to(device)
    
    logits = model(batch)
    
    probs = F.softmax(logits, dim=1)
    # probs = logits
    return probs.detach().cpu().numpy()

store_ious, store_ious_cls = evaluate_explanation(model, xai_dataloader, xai_dataset_size, batch_predict, save_explanations_path, expl_thr=0.2, device_id = device_id)

np.save('store_ious.npy', store_ious)
with open('store_ious_cls.pkl', 'wb') as f:
    pickle.dump(store_ious_cls, f)