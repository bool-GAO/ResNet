import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from torch.autograd import Variable




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 50))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))
    return ave_loss


def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('TestSet: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        return avgloss

if __name__ == '__main__':
    # 设置超参数
    BATCH_SIZE = 16
    EPOCHS = 2000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((200, 300))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    # 读取数据
    dataset_train = datasets.ImageFolder('C:/Users/gsx66/Desktop/classfication/sustao/train', transforms)
    dataset_test = datasets.ImageFolder('C:/Users/gsx66/Desktop/classfication/sustao/test', transforms)

    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True)
    modellr = 1e-4

    # 实例化模型并且移动到GPU
    criterion = nn.CrossEntropyLoss()
    model = torchvision.models.resnet18(weights=None)  # ****
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model.to(DEVICE)
    # 选择简单暴力的Adam优化器，学习率调低
    optimizer = optim.Adam(model.parameters(), lr=modellr)
    # 训练
    min_loss = 100
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        loss_trian = train(model, DEVICE, train_loader, optimizer, epoch)
        loss_test = val(model, DEVICE, test_loader)
        if loss_test < min_loss:
            print('save ,model...')
            torch.save(model, './model.pth')
            min_loss = loss_test
            # 保存网络位TORCHSCRIPT
            dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
            traced_cell = torch.jit.trace(model, dummy_input)
            traced_cell.save('./jit_model.pt' )