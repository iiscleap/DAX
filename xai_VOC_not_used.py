import torch
# import torchvision
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision.datasets as datasets
# import torch.utils.data as data
import torchvision.transforms as transforms
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# import torchvision.models as models
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import Saliency, IntegratedGradients, LayerGradCam
from captum.attr import NoiseTunnel
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import sys

import explain_VOC as DAME

from pdb import set_trace as bp

def get_annt_transform(shape): 
    assert len(shape)==2
    transf = transforms.Compose([
        transforms.Resize(shape),
        transforms.CenterCrop(shape[0])
    ])    

    return transf

def evaluate_explanation(model, xai_dataloader, xai_dataset_size, batch_predict, save_explanations_path, expl_thr, device_id):
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    # since = time.time()
    if not os.path.exists(save_explanations_path): os.mkdir(save_explanations_path)
    print('...Evaluating explanations...')

    running_loss = 0.0
    running_corrects = 0
    miss_count = 0
    store_ious = []
    store_ious_cls = {x: [] for x in range(21)}
    # Iterate over data.
    for i, (inputs, labels, annts) in enumerate(xai_dataloader):
        if i%100==0: print('iter:', i)
        model.eval()
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        # weights = torch.tensor([torch.count_nonzero(labels==i)  for i in range(21)]).to(device)
        # weights = weights/weights.sum()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        assert preds.shape[0] == 1

        if preds[0] == labels[0]:
            # int_grad = LayerGradCam(model, model.layer4[-1])
            # # vanilla_gradient = Saliency(model)
            # # noise_tunnel = NoiseTunnel(vanilla_gradient)
            # # attributions_ig = int_grad.attribute(inputs, nt_samples=30, nt_type='smoothgrad', target=labels)
            # # attributions_ig = noise_tunnel.attribute(img, nt_samples=30, nt_type='smoothgrad', target=labels, abs=False)
            # attributions_ig = int_grad.attribute(inputs, target=labels)
            # upsample = torch.nn.Upsample(size=(inputs.shape[-2], inputs.shape[-1]), mode='bilinear')
            # attributions_ig = upsample(attributions_ig).squeeze(0)
            # # attributions_ig = attributions_ig.mean(axis=1)

            # #......alternate implementation.........
            # cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]], use_cuda=True)
            # attributions_ig = cam(input_tensor=inputs, targets=[ClassifierOutputTarget(labels.item())])
            # attributions_ig = torch.from_numpy(attributions_ig)
            # #.......................................
            attributions_ig = DAME.generate_explanation(inputs, 
                                         batch_predict, # classification function
                                         top_labels=1, 
                                         hide_color=None,
                                         batch_size=256, 
                                         num_samples=3000, idx_expl = labels.item(), device_id=device_id, random_seed=42) # number of images that will be sent to classification function

            # temp, attributions_ig = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=100, hide_rest=False)
            attributions_ig = (attributions_ig-attributions_ig.min())/(attributions_ig.max()-attributions_ig.min())

            save_expl_cls_path = os.path.join(save_explanations_path, str(labels[0].item()))
            if not os.path.exists(save_expl_cls_path): os.mkdir(save_expl_cls_path)
            np.save(os.path.join(save_expl_cls_path, 'mask_'+str(i)+'.npy'), attributions_ig)

            attributions_ig = torch.from_numpy(attributions_ig).unsqueeze(0)
            
            mask = torch.zeros(attributions_ig.shape)
            attributions_ig = (attributions_ig - attributions_ig.min())/(attributions_ig.max()-attributions_ig.min())
            mask[attributions_ig>expl_thr] = 1
            mask = mask.squeeze(0)
            assert len(mask.shape) == 2

            annt_transf = get_annt_transform(mask.shape)
            annt_mask = torch.zeros(annts.shape)
            annt_mask[annts==labels[0].item()] = 1
            annt_mask = annt_transf(annt_mask).squeeze(0)
            annt_mask[annt_mask>=0.4] = 1
            annt_mask[annt_mask<0.4] = 0

            mask_iou = mask+annt_mask
            iou = torch.sum(mask_iou==2)/torch.sum(mask_iou>=1)
            print(f'IoU: {iou}')
            store_ious.append(iou)
            store_ious_cls[labels[0].item()].append(iou)
            sys.stdout.flush()
            sys.stderr.flush()
        else: miss_count+=1
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    print(f'miss classfication percentage that shd not have happened: {miss_count*100/(i+1)}')
    print(f'IoU stats: mean- {np.mean(store_ious)}, std- {np.std(store_ious)}')
    print(f'classwise IoU values: {[[np.mean(store_ious_cls[c]), np.std(store_ious_cls[c])] for c in range(21)]}')
    return np.array(store_ious), store_ious_cls

