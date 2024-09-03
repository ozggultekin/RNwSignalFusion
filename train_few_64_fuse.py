# code is based on https://github.com/floodsung/LearningToCompare_FSL
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
import time

import warnings 
warnings.filterwarnings('ignore') 

torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
#torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
#torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.is_available()
torch.cuda.set_device("cuda:0")

parser = argparse.ArgumentParser(description="Training Process for RN with Sensor Fusion")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 10)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 20)
parser.add_argument("-e","--episode",type = int, default= 300)
parser.add_argument("-t","--test_episode", type = int, default = 300)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int, default=10)
parser.add_argument("-ft","--file_type",type=str, default="_vibCon_SEU_img")
args = parser.parse_args()


# Hyperparameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
FILE_TYPE = args.file_type

max_accuracy = 0.0

print(f"{CLASS_NUM} way {SAMPLE_NUM_PER_CLASS} shot {FILE_TYPE}")

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layerC2 = nn.Sequential(nn.AvgPool2d((1,2), stride=(1,2)))
        self.layerM2 = nn.Sequential(nn.MaxPool2d((1,2), stride=(1,2)))
        self.layerC3 = nn.Sequential(nn.AvgPool2d((1,3), stride=(1,3)))
        self.layerM3 = nn.Sequential(nn.MaxPool2d((1,3), stride=(1,3)))
        
    def forward(self,x):
        out = self.layer1(x[:,:,:,:64])
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if (x.shape[3] == 128):
            out2 = self.layer1(x[:,:,:,64:128])
            out2 = self.layer2(out2)
            out2 = self.layer3(out2)
            out2 = self.layer4(out2)
            outC = torch.cat((out,out2), axis=3)
            outA = self.layerC2(outC)
            outM = self.layerM2(outC) 
            out = outA+outM
        elif (x.shape[3] == 192):
            out2 = self.layer1(x[:,:,:,64:128])
            out2 = self.layer2(out2)
            out2 = self.layer3(out2)
            out2 = self.layer4(out2)
            out3 = self.layer1(x[:,:,:,128:])
            out3 = self.layer2(out3)
            out3 = self.layer3(out3)
            out3 = self.layer4(out3)
            outC = torch.cat((out,out2,out3), axis=3)
            outA = self.layerC3(outC)
            outM = self.layerM3(outC) 
            out = outA+outM
            
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    metatrain_folders, metatest_folders = tg.RNwFuse_folders(FILE_TYPE)

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    global max_accuracy
    if iteration == 0:
        max_accuracy == 0
    else:
        max_accuracy = max_accuracy

    for episode in range(1,EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        task = tg.RNwFuseTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_RNwFuse_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_RNwFuse_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)

        # sample datas
        samples,sample_labels = next(iter(sample_dataloader))
        batches,batch_labels = next(iter(batch_dataloader))

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU))
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,14,14)
        sample_features = torch.sum(sample_features,1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda(GPU))

        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,14,14)
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).cuda(GPU))
        loss = mse(relations,one_hot_labels)

        # training
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if episode%30 == 0:

            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                task = tg.RNwFuseTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,15)
                sample_dataloader = tg.get_RNwFuse_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 5
                test_dataloader = tg.get_RNwFuse_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)

                sample_images,sample_labels = next(iter(sample_dataloader))
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]

                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU))
                    sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,14,14)
                    sample_features = torch.sum(sample_features,1).squeeze(1)
                    test_features = feature_encoder(Variable(test_images).cuda(GPU))

                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)

                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,14,14)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)


                accuracy = total_rewards/1.0/CLASS_NUM/15
                accuracies.append(accuracy)


            test_accuracy, h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)
            logs.writelines("episode:" + str(episode) + " test accuracy:" + str(round(test_accuracy,4)) + "\n")
            if test_accuracy > last_accuracy:

                # save networks
                torch.save(feature_encoder.state_dict(),str("./models/" + now + "_RNwFuse_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot" + FILE_TYPE + ".pkl"))
                torch.save(relation_network.state_dict(),str("./models/" + now + "_RNwFuse_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot" + FILE_TYPE + ".pkl"))

                print("save networks for episode:",episode)
                logs.writelines("saved...\n")
                last_accuracy = test_accuracy
                if last_accuracy > max_accuracy:
                    max_accuracy = last_accuracy
        
    logs.write("\nLast maximum accuracy: " + str(round(last_accuracy,4))+ "\n\n\n")
    print("\nLast maximum accuracy: " + str(round(max_accuracy,4))+ "\n\n")

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    logs = open(r"./Logs/Train/" + now + " " + FILE_TYPE + " " + str(CLASS_NUM) + "w" + str(SAMPLE_NUM_PER_CLASS) + "s.txt", 'w')
    for iteration in range (50):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs.writelines(FILE_TYPE + ": " + str(CLASS_NUM) + " way " + str(SAMPLE_NUM_PER_CLASS) + " shot\n" + now + "\n---\n")
        print("\n" + str(iteration))
        main()
    logs.write("---\nMaximum accuracy: " + str(round(max_accuracy,4)) + "\n\n\n")
    time.sleep(1)
    print("\nMaximum accuracy: "+ str(round(max_accuracy,4)))
    logs.close()