from PIL import Image
from torchvision import models, transforms
from matplotlib import pyplot as plt
from torch import nn
import os
import numpy as np
import torch.nn.functional as F
from pdb import set_trace as bp

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)


def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_jitter_transform(): 
    transf = transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=.3, saturation = 0.5, contrast = 0.4)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()
jitter_transform = get_jitter_transform()

bp()
img = get_image('../ant.jpeg')
img = pill_transf(img)
img = jitter_transform(img)
plt.imsave('img1.png', np.array(np.array(img)))