# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time
import copy

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from ENet import ENet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ENet(20)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    # state_dict = torch.load('model_best.pth')['state_dict']
    state_dict = torch.load('state_dict15.pth')
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
        
    model.load_state_dict(new_state_dict)
    print ("Model and weights LOADED successfully")
    
    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(20)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)
        # print(labels.shape)
        # print(outputs.shape)
        # print(outputs.max(1)[1].unsqueeze(1).shape)
        iouEvalVal. tor(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    # print("TOTAL IOU: ", iou * 100, "%")
    print(iou_classes)
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="./")
    parser.add_argument('--loadWeights', default="state_dict15.pth")
    parser.add_argument('--loadModel', default="ENet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="../../ERF_Net/train/leftImg8bit_trainvaltest")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
