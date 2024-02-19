# Copyright (c) OpenMMLab. All rights reserved.
import os
# import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from ENet import ENet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import copy
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="")
    parser.add_argument('--loadWeights', default="state_dict15.pth")
    parser.add_argument('--loadModel', default="ENet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="../../ERF_Net/train/leftImg8bit_trainvaltest")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ENet(20)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    #     own_state = model.state_dict()
    #     for name, param in state_dict['state_dict'].items():
    #         if name not in own_state:
    #             if name.startswith("module."):
    #                 own_state[name.split("module.")[-1]].copy_(param)
    #                 # print(param)
    #                 # print('of')
    #             else:
    #                 if not name.startswith("module."):
    #                     k = "module."+str(name)
    #                     own_state[k].copy_(param)
    #                 else:
    #                     print(name, " not loaded")
    #                 # print('bok')
    #                     continue
    #         else:
    #             own_state[name].copy_(param)
    #     return model

    # model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    # state_dict = torch.load('model_best.pth')['state_dict']
    state_dict = torch.load("state_dict25.pth")
    prefix = 'module.'

# Create a new state dictionary with updated keys
    updated_state_dict = {f"{prefix}{key}": value for key, value in state_dict.items()}

    def replace_batchnorm_keys(original_key):
    # Split the key into parts
        parts = original_key.split('.')
        new_key=original_key
        if parts[2]=="batchnorm" and parts[1] != "init":
            new_key = parts[0]+'.'+parts[1]+'.'+parts[2]+"1"+'.'+parts[3]
        elif parts[2]=="batchnorm2":
            new_key = parts[0]+'.'+parts[1]+'.'+"batchnorm3"+'.'+parts[3]
        # Check if the key contains 'batchnorm' followed by a number
        # for i, part in enumerate(parts):
        #     if part.startswith('batchnorm'):
        #         # Extract the number after 'batchnorm'
        #         number = part[len('batchnorm'):]
                
        #         # Replace 'batchnorm' with 'batchnorm1' or 'batchnorm2' with 'batchnorm3', and so on
        #         new_part = f'batchnorm{int(number) + 1}'
                
        #         # Replace the original part in the list
        #         parts[i] = new_part
        
        # # Join the modified parts back into a key
        # new_key = '.'.join(parts)
        
        return new_key
    
# Create a new state dictionary with updated batchnorm keys
    updated_state_dict = {replace_batchnorm_keys(key): value for key, value in updated_state_dict.items()}
    # updated_state_dict = {replace_batchnorm1_keys(key): value for key, value in updated_state_dict.items()}
    new_state_dict = copy.deepcopy(updated_state_dict)
    for key,value in updated_state_dict.items():
        parts = key.split('.')
        new_key=key
        if parts[2]=="batchnorm1" and parts[1] != "init":
            new_key = parts[0]+'.'+parts[1]+'.'+"batchnorm2"+'.'+parts[3]
            new_state_dict[new_key]= value
        
    # print(updated_state_dict['module.b11.batchnorm3.bias'].shape)
    # print(model.state_dict()['module.b11.batchnorm3.bias'].shape)

    model.load_state_dict(new_state_dict)
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
            # print(result.shape) #[1,20,1080,1920]
            # print(result)
        anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)            
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts) #replacing all the values of '2' with values of 1 and leaving the other unchanged
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    # print(val_out)
    # print(val_label)
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()