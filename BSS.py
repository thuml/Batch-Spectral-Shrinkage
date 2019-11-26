import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torch.nn as nn
import model as model_no
import transform as trans
import argparse
import os
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='PyTorch finetune experiment')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--trainpath', type=str, default='./', metavar='S',
                        help='path of training dataset')
parser.add_argument('--testpath', type=str, default='./', metavar='S',
                        help='path of testing dataset')
parser.add_argument('--method', type=str, default='l2', metavar='S',
                        help='method:l2 or l2+bss')
parser.add_argument('--lr', type=float, default=0.01,
                        help='init learning rate')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

data_transforms = {
    'train': trans.transform_train(resize_size=256, crop_size=224),
    'val': trans.transform_train(resize_size=256, crop_size=224),
}
data_transforms = trans.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)

# set dataset
batch_size = {"train": 48, "val": 100, "test": 100}
for i in range(10):
    batch_size["val" + str(i)] = 4


trainpath = args.trainpath
testpath = args.testpath

dsets = {"train": datasets.ImageFolder(root=trainpath, transform=data_transforms["train"]),
         "val": datasets.ImageFolder(root=testpath, transform=data_transforms["val"]),
         "test": datasets.ImageFolder(root=testpath, transform=data_transforms["val"])}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=4)

for i in range(10):
    dsets["val" + str(i)] = datasets.ImageFolder(root=testpath,
                                                 transform=data_transforms["val" + str(i)])
    dset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dsets["val" + str(i)],
                                                               batch_size=batch_size["val" + str(i)], shuffle=False,
                                                               num_workers=4)
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val'] + ["val" + str(i) for i in range(10)]}

dset_classes = range(len(dsets['train'].classes))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def test_target(loader, model, test_iter=0):
    with torch.no_grad():
        start_test = True
        if test_iter > 0:
            iter_val = iter(loader['test'])
            for i in range(len(iter(loader['test']))):
                data = iter_val.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
        else:
            iter_val = [iter(loader['val'+str(i)]) for i in range(10)]
            for i in range(len(loader['val0'])):
                data = [iter_val[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].to(device)
                labels = labels.to(device)
                outputs = []
                for j in range(10):
                    output = model(inputs[j])
                    outputs.append(output)
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy



def ft_lr_scheduler(param_lr, optimizer, iter_num, iter=6000, new=0.1, init_lr=0.01):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * new**(iter_num//iter)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return optimizer



class Net(nn.Module):
    def __init__(self,num_features):
        super(Net,self).__init__()
        self.model_fc = model_no.Resnet50Fc()
        self.classifier_layer = nn.Linear(num_features, len(dset_classes))
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
    def forward(self,x):
        BSS = 0
        feature = self.model_fc(x)
        u,s,v = torch.svd(feature.t())
        ll = s.size(0)
        if args.method == 'l2+bss':
            for i in range(1):
                BSS = BSS + torch.pow(s[ll-1-i],2)
        else:
            BSS = 0
        outC = self.classifier_layer(feature)
        return(outC, BSS)



num_features = 2048
ResNet = Net(num_features)
ResNet = ResNet.to(device)

ResNet.train(True)
criterion = {"classifier":nn.CrossEntropyLoss()}
optimizer_dict =[{"params":filter(lambda p: p.requires_grad,ResNet.model_fc.parameters()), "lr":0.1},{"params":filter(lambda p: p.requires_grad,ResNet.classifier_layer.parameters()), "lr":1}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005)
train_cross_loss = train_transfer_loss = train_total_loss = train_sigma =0.0
len_source = len(dset_loaders["train"]) - 1
len_target = len(dset_loaders["val"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])
for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])
test_interval = 500
num_iter = 9002
torch.set_printoptions(threshold=10000000)
for iter_num in range(1, num_iter+1):
    ResNet.train(True)
    optimizer = ft_lr_scheduler(param_lr, optimizer, iter_num, iter=6000, new=0.1, init_lr=args.lr)
    optimizer.zero_grad()
    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["train"])
    data_source = iter_source.next()
    inputs, labels = data_source
    inputs = inputs.to(device)
    labels = labels.to(device)
    outC,BSS = ResNet(inputs)
    classifier_loss = criterion["classifier"](outC, labels)
    total_loss = classifier_loss+0.001*BSS
    total_loss.backward()
    optimizer.step()
    train_cross_loss += classifier_loss.item()
    train_total_loss += total_loss.item()
    if iter_num % test_interval == 0:
        print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average Transfer Loss: {:.4f};  Average Training Loss: {:.4f}".format(
            iter_num, train_cross_loss / float(test_interval), train_transfer_loss / float(test_interval),
            train_total_loss / float(test_interval)))
        train_cross_loss = train_transfer_loss = train_total_loss = 0.0
    if (iter_num % 9000) == 0:
        ResNet.eval()
        test_acc = test_target(dset_loaders, ResNet.predict_layer)
        print('test_acc:%.4f'%(test_acc))





