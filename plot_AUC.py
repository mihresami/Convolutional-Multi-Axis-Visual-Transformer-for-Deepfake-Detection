from cProfile import label
from turtle import color
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error,confusion_matrix
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import itertools
import torch.nn.functional as F
# from OursC2.ST_pyramid1 import ST_FPNet
from tqdm import tqdm
from torchvision import transforms
from dataProcess import *
import logging
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import argparse
# from EVitNet import ResvitNet,landVit
# from ResvitAttenNet  import RvitattenNet
# from resnet_18.Rsnet import RSNet
# from xception.xception import imagenet_pretrained_xception
# from VitTempNet import vitTNet
# from ResvitAttenNet import RvitattenNet
# from ImpvitNet import vitNet
# from mVit import M_vit
# from MaxVit import maxvit_t
from DeepModel import maxvit_t
# from ST_net import ST_NET
# from Cvit import CViT
# from vit import VitNet
# from swimNet import Swim_Net
# from RFreqencyNet import RF_Net
# from RNet import Res_Net
# from deepModel import WindowAttention_Net
# from EVitNet import ResvitNet


def plot_auc(args):
        

    trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    # trans_x = transforms.Compose([
    #     transforms.Resize([299, 299]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    # )
    # trans_i = transforms.Compose([
    #     transforms.Resize([299, 299]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # )
    # dataset = VideoDataset(args.data_path, transform=trans)
    dataset  = ImageDataset(args.data_path,transform=trans)
    # dataset = LandmarkDataset(args.data_path)
    train_size = int(len(dataset)*0.95)
    # valid_size = int(len(dataset)*0.15)
    valid_size = len(dataset) - train_size
    train_set,valid_set = data.random_split(dataset, [train_size, valid_size])
    test_loader = data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # print(len(dataset))
    print("Evaluation_Dataset",len(valid_set))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = S_Net(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers)
    # model = vitTNet(seq_length=30,hidden_dim=512,layers_dim=[512,49])
    # model = landVit(30,136)
    # model = RvitattenNet(196,768,[784,49])
    # model = ResvitNet(196,768)
    # model = vitTNet(196,768,layers_dim=[784,49])
    # model = vitNet(256,1280)
    # model = M_vit()
    model = maxvit_t()
    # model = VitNet()
    # model = CViT()
    # model = Swim_Net()
    # model = RF_Net(seq_length=196,hidden_dim=768)
    # model = Res_Net()
    # model = ST_NET()
    # model = ST_FPNet(128,3)
    # # model = WindowAttention_Net()
    # model.load_state_dict(torch.load('STNET_weight/ST_NET_FF++-10-acc0.8375568551007148.pt'), strict=False)
    # model = RvitattenNet(seq_length=196,hidden_dim=768,layers_dim=[784,49])
    model.load_state_dict(torch.load('model_weight\DeepModel_best_Frequency_all(best)-15-acc0.9828921357332897.pt'),strict=False)
    # model.load_state_dict(torch.load('OursC2/weight_kloss_88/CDF-99-acc0.9726666666666667.pt'), strict=False)
    #urNet/weight_all_data/DeepModel_frequency_all(FF_CDF_DF)-27-acc0.9760435101653213.pt
    #weight_all_data/DeepModel_frequency_all(FF_CDF_DF)-29-acc0.9921871627746364.pt
    # model.load_state_dict(torch.load('weight_deepmodel_freq/DeepModel_frequency_best(multiply_only)_all(best)-13-acc0.9757130754024287.pt'), strict=False)
    # model.load_state_dict(torch.load('weight_deepmodel_freq/DeepModel_freq(local(allmvit)_and_global(allRsNet)_and_Catlast_layersboth)_alldata-26-acc0.9943025920785048.pt'), strict=False)
    #1. OurNet/weight_deepmodel_freq/DeepModel_freq(local(allmvit)_and_global(allRsNet)_and_Catlast_layersboth)_alldata-26-acc0.9943025920785048.pt
    #2. OurNet/weight_deepmodel_freq/DeepModel_freq(local(allmvit)_and_global(allRsNet)_and_Catlast_layersboth)_alldata-17-acc0.9928529839883552.pt FF++C40 is not good
    # weight_deepmodel_freq/DeepModel_bestFreq_all-18-acc0.9927401601848527.pt FF++C40 auc = 88 and ACC 73
    # weight_deepmodel_freq/DeepModel_bestFreq_all_dfdc-15-acc0.9855862386216317.pt FF++c40 = 91.69
    #Best model weight-FF/RvitFFnewattention2only-12-acc0.9955428571428572.pt
    #S_net SNET/weight/CDF-63-acc0.830739299610895.pt
    #1 FF_4...the last layer is 4. 
    # Method 2 best AUC on kenan CDF weight-FF/RvitFFnewattention2only-12-acc0.9955428571428572.pt layer2 AUC = 71.51 
    # Method 2 best AUC and ACC on CDF weight-FF/RvitFFnewattentTF-19-acc0.9963047619047619.pt AUC = 70
    #Method 2 best AUC on compressed both weight_grad/RvitFFnewattention2-20-acc0.9935238095238095.pt
    # Method 2 best OurNet/weight_grad/RvitFFnewattention2-12-acc0.990247619047619.pt AUC = 74 72 48
    # Method2 best AUC and ACC on FF++ OurNet/weight-FF/RvitFFnewattentFF++PretrainedonTFbest-9-acc0.9906412478336222.pt
    # Method=2 atten 2 and 4 OurNet/weight_grad/RvitFFnewattention24-9-acc0.9816380952380952.pt
    model = model.to(device)

    # model.eval()
    
    prob_all = []
    
    label_all = []
    true_all = []
    # loggiing setting

    logging_name = 'Method2'
    log_path = 'log' + '/' + logging_name + 'test.txt'
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.INFO)
    handler  = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    with torch.no_grad():
        model.eval()
        # for i, ((X,L), y) in enumerate(tqdm(test_loader)):
        for i, (X, y) in enumerate(tqdm(test_loader)):
            # X,L,y = X.to(device), L.to(device), y.to(device)
            X,y = X.to(device),y.to(device)
            # y = y.unsqueeze(1)
            output = model(X)
            # if not isinstance(output, torch.Tensor):
            #     output = torch.sigmoid(output)
            # output = output.permute(1,0).squeeze(0)
            # # print(output.shape)
            # y_pred = output.clone()
            # print("label",y, "\n output",output)
            # predicted_labels = (y_pred > threshold).long()
            # print("predicted labels",predicted_labels,'\t output',y_pred)
            # prob_all.extend(output.cpu().detach().numpy())
            # true_all.extend(predicted_labels.cpu().detach().numpy())
            prob_all.extend(output[:,0].cpu().detach().numpy())
            true_all.extend(torch.max(output,1)[1].cpu().detach().numpy())
            label_all.extend(y.cpu().detach().numpy())
            y_preds = torch.max(output, 1)[1]
            tp += ((y_preds == 1) & (y == 1)).sum().item()
            tn += ((y_preds == 0) & (y == 0)).sum().item()
            fn += ((y_preds == 0) & (y == 1)).sum().item()
            fp += ((y_preds == 1) & (y == 0)).sum().item()
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    recall = 0
    if tp + fn != 0:
        recall = tp / (tp + fn)
    # f1 = 0
    # if precision + recall != 0:
    #     f1 = 2 * precision * recall / (precision + recall)
    
    # prob_all = np.array(prob_all)
    # label_all = np.array(label_all)
    # print("AUC:{:.4f}".format(roc_auc_score(label_all,prob_all)))

    cm = confusion_matrix(label_all,true_all)
    fpr, tpr, thresholds = roc_curve(label_all, prob_all,pos_label=0)
    auc_score = roc_auc_score(label_all,prob_all)
    f1_scores = f1_score(label_all,true_all)
    auc1 = auc(fpr,tpr)
  
    print('-----auc-----',auc1)
    # print('------roc auc score-----',auc_score)
    print(f"----f1 score------ {f1_scores:.5f}")
    print('----- confustion matrix ',cm)
    msr = mean_squared_error(label_all,true_all)
    print('-----Mean Square Rate ----',msr)
    # plt.figure()
    # lw = 2
    # # plt.plot(fpr,tpr,color="darkorange",lw=lw,label = "(AUC = %0.2f)" % auc_score)
    # plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {auc1:.4f}')
    # # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="---")
    # plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic")
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig('auc' + '/' + 'AUC on_'+ args.name + '.png', dpi=600)
    # plt.figure()
    # plt.imshow(cm, cmap=plt.cm.Blues)
    # row_sums = np.sum(cm, axis=1)
    # # Normalize rows to percentages
    # cm_percentage = cm / row_sums[:, np.newaxis] * 100
    # threshold = cm_percentage.mean()  # Adjust threshold as needed
    # text_color = ['black' if x > threshold else 'white' for x in cm_percentage.flatten()]
    # # Set class labels
    # class_labels = ['Fake', 'Real']
    # for i, j in itertools.product(range(cm_percentage.shape[0]), range(cm_percentage.shape[1])):
    #     text = f"{cm_percentage[i, j]:.1f}%"  # Format as percentage with one decimal
    #     if i == j:
    #         col = 'white'
    #     else:
    #         col = 'black'
    #     plt.text(j, i, text, ha='center', va='center', color=col)  # Adjust text color for contrast

    # # Set labels for rows and columns using class labels
    # plt.xticks(range(len(cm[0])), class_labels, rotation=45, ha='right')
    # plt.yticks(range(len(cm)), class_labels)
    # # Set title
    # plt.title("Confusion Matrix (Percentage Distribution)")

    # plt.tight_layout()
    # plt.show()
    # # Add axis labels and title
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig('auc' + '/' + args.name + "_Confustion matrix" '.png', dpi=600)
    # fpr = np.array(fpr)
    # tpr = np.array(tpr)
   
    # logger.info(args.name + ' TP: %d  TN: %d  FN: %d  FP: %d  Accuracy: %.5f AUC: %.5f MSE: %.5f  Precision: %.5f  Recall: %.5f f1_score: %.5f' % 
    #             (tp, tn, fn, fp, acc,auc1,msr, precision, recall,f1_scores))
    print(' TP: %d  TN: %d  FN: %d  FP: %d  Accuracy: %.6f AUC: %.6f MSE: %.6f Precision: %.6f  Recall: %.6f f1_score: %.6f' % 
          (tp, tn, fn, fp, acc, auc1,msr,precision, recall,f1_scores))
    # process_data = np.array([fpr,tpr])
    # np.savetxt(args.path + '/' + args.name + '_ROC_curve_data.txt', process_data, fmt='%d')
    # logger.info('My_auc ' + DATA_PATH[5:] + ' AUC: {}'.format(auc_score))
    # plot

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
    description='Evaluation metrics.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-hd', '--hidden_dim', type=int, default=128,
                        help="Input hidden_dim")
    parser.add_argument('-hl', '--hidden_layers', type=int, default=3,
                        help="Input hidden_layers")
    parser.add_argument('-e', '--epoches', type=int, default=100,
                        help="Input epoches")
    parser.add_argument('-bs','--batch_size',default=8,type=int)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help="Input learning_rate")
    parser.add_argument('-dp', '--data_path', type=str, default='dataset/dataset',
                        help="Input data_path")
    parser.add_argument('-p','--path',default='log',type=str)
    parser.add_argument('-n','--name',default='Method2_NT_C23(85%)',type=str)
    args = parser.parse_args()
    plot_auc(args)
    
