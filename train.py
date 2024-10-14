import os

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from torchvision import transforms
# from Dataprocess.dataProcess import VideoDataset1,ImageDataset,VideoDataset,VideoDatasetLand
from dataProcess import *
import logging
import argparse
import torch,gc
import torch.nn.functional as F
# from Cvit import CViT
# from ST_net import ST_NET
# from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
# from torch.utils.tensorboard import EarlyStopping

# from ResvitAttenNet import RvitattenNet
# from ImpvitNet import vitNet
# from mVit import M_vit
# from MaxVit import maxvit_t
# from vit import VitNet
# from swimNet import Swim_Net
from DeepModel import maxvit_t
# from RFreqencyNet import RF_Net
# from VitTempNet import vitTNet
# from DeepModel import maxvit_t
# from deepModel import WindowAttention_Net

# checkpoint = "google/vit-base-patch16-224-in21k"

train_tp = []
train_tn = []
train_fp = []
train_fn = []
valid_tp = []
valid_tn = []
valid_fp = []
valid_fn = []

best_acc = 0.00
import math
import numpy as np
def parse_parameter():
    parser = argparse.ArgumentParser(
        description='Train models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-hd', '--hidden_dims', type=int, default=128,
                        help="Input hidden_dims")
    parser.add_argument('-hl', '--hidden_layers', type=int, default=3,
                        help="Input hidden_layers")
    parser.add_argument('-e', '--epoches', type=int, default=15,
                        help="Input epoches")
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help="Input batch_size")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help="Input learning_rate")
    parser.add_argument('-dp', '--data_path', type=str, default= 'dataset\dataset'#'/home/spendar/data/DF/DeeperForensics''
                        ,help="Input data_path")
    parser.add_argument('-p','--path',type=str,default='Deepfake_detection')
    parser.add_argument('-n','--name',type=str,default='ST_NET-FF+CDF+DF')
    
    
    #CDFconcat = 36 dim from Resnet concatenated with input image 
    #CDF_novel88 using novel paper code....combiner is 256->128 output
    #kloss_88 is novel code with selection last 88. 2048 combiner output
    # FFvitandRes24attention two layer which are layer 2 and layer 4 are filtered by attention and linear layer is used to resize the size into 768
    # Note NewResatten2FFKenan++ this name is by mistake it is not kenan but Normal FF++ dataset and Resetn pretrained=False and Vit=True. Note
    #ResOnly is using resnet with output 512 with unsquueze(1) True and False
    # Resvit2NewFF is new version of best performing method but only positions are fixed. 768 and Linear = 768 output and outpt of model fc = 2
   
    args = parser.parse_args()
    return args

# loggiing setting
def init_logging(args):
    logging_name = args.name
    log_path = os.path.join('log', args.name + '.progress'+ '.txt')
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.INFO)
    handler  = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# data setting
def init_data(data_path, bs):
    # trans = transforms.Compose([
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    #     transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # )
#     transform_vgg =   transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
#         transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
    trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # trans_i = transforms.Compose([
    #     transforms.Resize([299, 299]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # )
    
    # Meso_trans = transforms.Compose([
    #     transforms.Resize([256, 256]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    # )
    # dataset = VideoDataset(data_path, transform=trans)
    dataset = ImageDataset(data_path, transform=trans)
    # dataset = LandmarkDataset(data_path)
    train_size = int(len(dataset)*0.75)
    valid_size = len(dataset) - train_size 
    train_set, valid_set = data.random_split(dataset, [train_size, valid_size])
    train_loader = data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=2,drop_last=True)
    valid_loader = data.DataLoader(valid_set, batch_size=bs, shuffle=True, num_workers=2,drop_last=True)

 
    return train_loader, valid_loader


def train(model, train_loader, logger, device, opt, criterion, e, args):
   
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_loss = 0.0
    total_pred = 0
    total_size = 0
    # total_batches = len(train_loader)
    
    for i, ((X, L), y) in enumerate(train_loader):
    # for i, (X,y) in enumerate(train_loader):
        model.train()
        batch_loss = 0.0
        batch_pred = 0
        loss = 0
        # threshold = 0.5
        
        X,L,y = X.to(device), L.to(device), y.to(device)
        # X, y = X.to(device), y.to(device)
        output = model(X,L)
      
        # Calculate contrastive loss within the loop (assuming separate labels for real/fake)
        # real_mask = (y == 1).unsqueeze(1)  # Create mask for real samples (batch_size, 1, feature_dim)
        # fake_mask = (y != 1).unsqueeze(1)  # Create mask for fake samples (batch_size, 1, feature_dim)

        # real_features = feature * real_mask  # Mask out fake features in real samples
        # fake_features = feature * fake_mask  

        # pairwise_distances = torch.nn.functional.pairwise_distance(real_features, fake_features, keepdim=True)
        # positive_distances = pairwise_distances[..., 0]
        # negative_distances = pairwise_distances
        # margin = 1
        # contrastive_loss = torch.clamp(margin + positive_distances - negative_distances, min=0.0)
        # contrastive_loss = torch.mean(contrastive_loss)
        loss = criterion(output,y)
        # ce_weight = 1
        # contrastive_weight = 0.5
        # total_loss = ce_weight * loss + contrastive_weight * contrastive_loss
        
        batch_loss += loss.item() 
        total_loss += loss.item() 
        total_size += output.size(0)
        y_pred = torch.max(output, 1)[1]
        batch_pred += y_pred.eq(y).sum().item()
        total_pred += y_pred.eq(y).sum().item()
        tp += ((y_pred == 1) & (y == 1)).sum().item()
        tn += ((y_pred == 0) & (y == 0)).sum().item()
        fn += ((y_pred == 0) & (y == 1)).sum().item()
        fp += ((y_pred == 1) & (y == 0)).sum().item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        logger.info('Epoch: %d/%d || batch: %d || avg loss: %.10f || acc: %.10f' % (e+1, args.epoches, i+1, batch_loss / args.batch_size, batch_pred / args.batch_size))
        

    global train_tp
    global train_fp
    global train_tn 
    global train_fn
    train_tp.append(tp)
    train_fp.append(fp)
    train_tn.append(tn)
    train_fn.append(fn)
    logger.info('Epoch: %d/%d  train acc: %.10f train loss: %.10f' % (e+1, args.epoches, total_pred / total_size,total_loss / total_size))


def eval(model, valid_loader, logger, device, criterion,e, args):
    
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    total_loss = 0.0
    total_pred = 0
    total_size = 0
    valid_loss = 0.0
    threshold = 0.5
    
    for i, ((X, L), y) in enumerate(valid_loader):
    # for i, (X, y) in enumerate(valid_loader):
        model.eval()
        with torch.no_grad():
            X, L,y = X.to(device),L.to(device), y.to(device)
            # X,y = x.to(device), y.to(device)
            # y = y.unsqueeze(1).float()
            output = model(X,L)
            total_size += output.size(0)
            loss = criterion(output, y)
            total_loss += loss.item()
            y_preds = torch.max(output, 1)[1]
            # print(y_preds,y)
            total_pred += y_preds.eq(y).sum().item()
            tp += ((y_preds == 1) & (y == 1)).sum().item()
            tn += ((y_preds == 0) & (y == 0)).sum().item()
            fn += ((y_preds == 0) & (y == 1)).sum().item()
            fp += ((y_preds == 1) & (y == 0)).sum().item()
    
    global valid_tp
    global valid_fp
    global valid_tn
    global valid_fn
    valid_tp.append(tp)
    valid_fp.append(fp)
    valid_tn.append(tn)
    valid_fn.append(fn)
    acc = total_pred / total_size
    valid_loss = total_loss / total_size
    # scheduler.step(valid_loss)
    # if early_stopping.early_stop:
    #     print("Early stopping triggered!")
    #     break
    
    logger.info('Epoch: %d/%d  valid acc: %.10f valid loss: %.10f' % (e+1, args.epoches, acc,valid_loss))
    global best_acc
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), 'model_weight'+ '/{}-{}-acc{}.pt'.format(args.name, e+60, best_acc))
        #weight_kloss is Normal model with layer 32,16,8,8 with kenan Loss.
        # OurNet/weight_vitTNet 

# def adjust_learning_rate(optimizer, epoch,initial_lr,decay_rate):
#     if not epoch % 10:
#         lr = initial_lr * torch.exp(-decay_rate * epoch)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr


def save_data(args):
    global train_tp
    global train_fp
    global train_tn
    global train_fn
    global valid_tp
    global valid_fp
    global valid_tn
    global valid_fn
    train_tp = np.array(train_tp)
    train_tn = np.array(train_tn)
    train_fp = np.array(train_fp)
    train_fn = np.array(train_fn)
    valid_tp = np.array(valid_tp)
    valid_tn = np.array(valid_tn)
    valid_fp = np.array(valid_fp)
    valid_fn = np.array(valid_fn)
    process_data =np.array([train_tp, train_tn, train_fp, train_fn, valid_tp, valid_tn, valid_fp, valid_fn])
    

    filename = 'log' + '/' + args.name + '.txt'
    if os.path.exists(filename):
        with open(filename, 'a') as f:  # Open in append mode
            f.write("\n")  # Add a newline for separation (optional)
            np.savetxt(f, process_data, fmt='%d')
    else:
        np.savetxt(filename, process_data, fmt='%d')
    

if __name__ == '__main__':
    args = parse_parameter()
    logger = init_logging(args)
    train_loader, valid_loader = init_data(args.data_path, args.batch_size)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==============",device,"==============")
    # torch.autograd.set_detect_anomaly(True)
    # dataiter = iter(train_loader)
    # img,label = next(dataiter)
    # print(img.shape)
    # model = vitTNet(seq_length=196,hidden_dim=768,layers_dim=[512,49])
    # model = vitNet(256,1280)
    # model = M_vit()
    # model = maxvit_t()
    model = ST_NET()
    # model = CViT()
    # model = VitNet()
    # model  = Swim_Net()
    # model = RF_Net(seq_length=196,hidden_dim=768)
    # model = WindowAttention_Net()
    # model = RvitattenNet(seq_length=196,hidden_dim=768,layers_dim=[784,49])
    # model.load_state_dict(torch.load('weight_grad/RvitFFnewattention2-107-acc0.9793039181692095.pt'), strict=False)
    # model.load_state_dict(torch.load('STNET_weight/ST_NET_FF++-21-acc0.8810916179337231.pt'), strict=False)
    model.to(device)
    model.load_state_dict(torch.load("STNET_weight/ST_NET-FF++Kenan-56-acc0.921875.pt"),strict=False)
    # opt = torch.optim.Adam(model_parameters, lr=args.learning_rate)
    # model_param = list(model.vitmodel.parameters()) + list(model.mlp.parameters()) + \
    # list(model.classification.parameters()) 
    # + list(model.multi_atten.parameters())
    model_param = list(model.parameters())
    optimizer = torch.optim.SGD(model_param , lr=args.learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=None)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=4, factor=0.1,mode='min',verbose=True)
    # early_stopping = EarlyStopping(patience=5, verbose=True)

    for e in tqdm(range(args.epoches)):
        train(model, train_loader, logger, device, optimizer, criterion, e, args)
        eval(model, valid_loader, logger, device,criterion, e, args)
    save_data(args)
    

