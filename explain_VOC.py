import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import time
from pdb import set_trace as bp
import torch
import random
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries

#fix seed and set deterministic behavior
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

 # resize and take the center part of image to what our model expects
# def get_input_transform():
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])       
#     transf = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize
#     ])    

#     return transf

# def get_input_tensors(img):
#     transf = get_input_transform()
#     # unsqeeze converts single image to batch of 1
#     return transf(img).unsqueeze(0)

# def get_pil_transform(): 
#     transf = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224)
#     ])    

#     return transf

# def get_preprocess_transform():
#     # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                                 std=[0.229, 0.224, 0.225])     
#     # transf = transforms.Compose([
#     #     transforms.ToTensor(),
#     #     normalize
#     # ])   

#     transf_fn = torch.nn.Sequential(
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     )
#     transf = torch.jit.script(transf_fn) 

#     return transf 

def generate_explanation(img, batch_predict, top_labels=1, hide_color=None, batch_size=128, num_samples=3000, idx_expl = None, device_id=None, random_seed=42, idx_file=None):
    #GPU to be used
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    # image transforms for the explainer
    
    # pill_transf = get_pil_transform()
    # preprocess_transform = get_preprocess_transform()

    # test_pred = batch_predict([pill_transf(img)])
    # print(test_pred.squeeze().argmax())

    #scale to full
    img_torch = img.detach().clone()
    img = (img-img.min())/(img.max()-img.min())
    img = img*255

    #explainer
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8), 
                                            batch_predict, # classification function
                                            top_labels=1, 
                                            img_init = img_torch,
                                            hide_color=None, 
                                            batch_size = batch_size,
                                            num_samples=num_samples, idx_expl = idx_expl, device_id=device_id, random_seed=SEED, idx_file=idx_file) # number of images that will be sent to classification function
    return explanation