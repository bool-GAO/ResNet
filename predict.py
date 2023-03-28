import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

if __name__ == '__main__':
    # 设置超参数
    BATCH_SIZE = 16
    # 数据预处理
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((200, 300))
    ])
    # 读取数据
    dataset_test = datasets.ImageFolder('C:/Users/gsx66/Desktop/ClassifyNetwork/sustao/test', transforms)
    print(len(dataset_test))
    # 导入数据
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True)
    print(len(test_loader))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load('model.pth')
    model = model.to(device)
    testing_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = Variable(data).to(device), Variable(target).to(device)
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == target.data)
        print("Test Accuracy is:{:.4f}%".format(100*testing_correct/len(dataset_test)))