# coding:utf-8
'''
train video classifier and predict
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant_, xavier_uniform_
from torchvision import transforms
from model import Net
from dataset import VideoDataset
import numpy as np
import time, sys, random, os
import pandas as pd
from setting import FEATURE_SAVEPATH, LOAD_CHECKPOINT_PATH, MODEL_SAVE_DIR, TRAIN_CSV
from setting import NUM_EPOCHS, NUM_CLASSES, NEW_NUM_CLASSES, BATCH_SIZE, INPUT_SIZE, FRAME_SAMPLE_NUM, IMG_EXT, MULTI_GPU, LOAD_OPTIM

print("----------SETTING----------")
print("FEATURE_SAVEPATH =", FEATURE_SAVEPATH)
if FEATURE_SAVEPATH != "none":
    ext_feature = True
    print(">>>>>NO TRAINING")
else:
    ext_feature = False
print("LOAD_CHECKPOINT_PATH =", LOAD_CHECKPOINT_PATH)
print("MODEL_SAVE_DIR =", MODEL_SAVE_DIR)
print("TRAIN_CSV =", TRAIN_CSV)
print("NUM_EPOCHS={}, NUM_CLASSES={}, NEW_NUM_CLASSES={}, BATCH_SIZE={}, INPUT_SIZE={}".format(NUM_EPOCHS, NUM_CLASSES, NEW_NUM_CLASSES, BATCH_SIZE, INPUT_SIZE))
print("FRAME_SAMPLE_NUM={}, IMG_EXT={}, MULTI_GPU={}, LOAD_OPTIM={}".format(FRAME_SAMPLE_NUM, IMG_EXT, MULTI_GPU, LOAD_OPTIM))
print("---------------------------")

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


def init_weights(m):
    if type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    elif type(m) == nn.Conv3d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.)
    elif type(m) == nn.BatchNorm3d:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.)
    elif type(m) == nn.Linear:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def train_model(model, dataloaders, optimizer, num_epochs=30):
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    start_t = time.time()
    ext_feature = False
    if FEATURE_SAVEPATH != "none":
        print(">>>NO TRAIN, only feature extracting")
        phase_list = ['valid']
        ext_feature = True
        feature_dict = {}
        if MULTI_GPU:
            model.module.ext_feature = True
        else:
            model.ext_feature = True
    else:
        phase_list = ['train', 'valid']

    for epoch in range(num_epochs):
        print('>>>Train epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in phase_list:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.
            total_proc = 0

            for inputs, labels, video_info_list in dataloaders[phase]:
                if inputs.shape[0] != BATCH_SIZE and not ext_feature:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if ext_feature:
                        outputs_np = outputs.cpu().data.numpy()
                        for output, frame_dir in zip(outputs_np, video_info_list):
                            feature_dict[frame_dir] = output
                        continue

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_proc += inputs.size(0)


            if ext_feature:
                import pickle
                with open(FEATURE_SAVEPATH, 'wb') as f:
                    pickle.dump(feature_dict, f)
                print(">saved feature to", FEATURE_SAVEPATH)
                sys.exit()

            epoch_loss = running_loss / float(total_proc)
            epoch_acc = running_corrects.item() / float(total_proc)

            elapsed_min = round((time.time()-start_t)/60.0, 1)
            print('{} loss: {:.3f} accuracy: {:.3f} elapsed = {} min'.format(phase, epoch_loss, epoch_acc, elapsed_min))
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                savepath = MODEL_SAVE_DIR + "/ckpt_" + str(epoch)
                print(">>>best accuracy, checkpoint saving=", savepath)
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'valid_acc': epoch_acc
                            }, savepath)


print(">Dataset Loading")

df = pd.read_csv(TRAIN_CSV) # df.columns : frame_dir, start_frame, end_frame, label, train
df_train = df[df["train"]==True].reset_index(drop=True)
df_valid = df[df["train"]==False].reset_index(drop=True)
# todo rev # show sample csv

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ]),
}

train_dataset = VideoDataset(df_train, FRAME_SAMPLE_NUM, IMG_EXT, INPUT_SIZE, transform=data_transforms["train"])
valid_dataset = VideoDataset(df_valid, FRAME_SAMPLE_NUM, IMG_EXT, INPUT_SIZE, transform=data_transforms["valid"])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}



print(">Model Network Initializing")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_net = Net(NUM_CLASSES, BATCH_SIZE, FRAME_SAMPLE_NUM, INPUT_SIZE).to(device)
model_net.apply(init_weights)

if LOAD_CHECKPOINT_PATH != "none":
    print(">Trained model loading from", LOAD_CHECKPOINT_PATH)
    checkpoint = torch.load(LOAD_CHECKPOINT_PATH)
    model_net.load_state_dict(checkpoint['model_state_dict'])
    if LOAD_OPTIM:
        print(">optimizer loading from", LOAD_CHECKPOINT_PATH)
        optimizer_net.load_state_dict(checkpoint['optimizer_state_dict'])

if MULTI_GPU:
    model_net = torch.nn.DataParallel(model_net)

if NUM_CLASSES != NEW_NUM_CLASSES:
    if MULTI_GPU:
        model_net.module.final_layer = nn.Linear(512, NEW_NUM_CLASSES).to(device)
    else:
        model_net.final_layer = nn.Linear(512, NEW_NUM_CLASSES).to(device)

optimizer_net = optim.Adadelta(model_net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.)
'''optimizer_net = torch.optim.SGD(model_net.parameters(),
                                0.001,
                                momentum=0.9,
                                weight_decay=0.0005,nesterov=True)
'''

print("---optimizer---")
print(optimizer_net)

model_net = train_model(model_net, dataloaders_dict, optimizer_net, num_epochs=NUM_EPOCHS)
