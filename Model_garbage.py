import os
import random
import shutil
import torch
from PIL import Image
import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

import matplotlib.pyplot as plt
import shutil
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import timm
import time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# trực quan hóa dữ liệu
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parser = ArgumentParser(description="Garbage Xception")
    parser.add_argument("--epochs", "-e", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--logging", "-l", type=str, default="C:/pythonProject/Pycharm_pythoncode/tensorboard")
    parser.add_argument("--trained_models", "-t", type=str,default="C:/pythonProject/Pycharm_pythoncode/trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args
# Model
# Tải mô hình Xception với trọng số ImageNet
base_model = timm.create_model("legacy_xception",pretrained=True)  
# tải mô hình với trọng số lấy từ ImageNet
#, vì trong timm bây giờ đổi thành legacy_xception chứ không còn là xception
base_model.global_pool = nn.Identity()  # Loại bỏ GAP để giữ đầu ra 4D, vì mô hình sử dụng GAP để làm phẳng ở ngay sau CNN nên cần loại bỏ đi vì ở dưới
# mô hình CNN mà mình tạo để thỏa mãn số class của riêng mình sẽ yêu cầu đầu vào là mảng 4D.
base_model.fc = nn.Identity()  # Bỏ lớp cuối cùng (fully connected) Bỏ đi lớp fully connected để dùng làm feature extractor
# Đóng băng trọng số để không huấn luyện lại toàn bộ mô hình
for param in base_model.parameters():
    param.requires_grad = False
# Lấy số lượng kênh đầu ra của base_model
in_features = base_model.num_features

# Xây dựng mô hình
class XceptionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(XceptionModel, self).__init__()
        self.base = base_model  # lấy trọng số, và phân thân của model
        # cắm nó với lớp của mình
        # in_channels bằng 2048 vì kích thước phần thân của Xception khi ảnh đi qua sẽ tạo ra 2048 channels
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=512)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)  # Số lớp đầu ra

    def forward(self, x):
        x = self.base(x)  # trích xuất đặc trưng từ model

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.gap(x)  # Áp dụng GAP
        x = torch.flatten(x, 1)  # Chuyển (batch_size, 512, 1, 1) thành (batch_size, 512)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    args = get_args()
    # check thông tin data
    data_path = "C:/pythonProject/Pycharm_pythoncode/garbage_classification"
    new_data_path = "C:/pythonProject/Pycharm_pythoncode/garbage_classification_merge"

    '''
    bây giờ từ những tấm ảnh đã có ta cần tạo ra 1 file train, test, value 
    '''
    # tạo thư mục cho train test và value
    mainPath = "C:/pythonProject/Pycharm_pythoncode"
    train_dir = mainPath + "/Dataset/train"
    val_dir = mainPath + "/Dataset/val"
    test_dir = mainPath + "/Dataset/test"

    '''
    +) Tiền xử lý ảnh trước khi đưa vào deeplearning: ta sử dụng transform 
    +) Dùng ImageFoder để load ảnh cũng như gán labels cho bức ảnh
    +) Dùng DataLoader để đóng gói 
    '''
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize về kích thước chuẩn
        transforms.RandomRotation(20),  # xoay ảnh 20 độ
        transforms.RandomHorizontalFlip(),  # Lật ngang ảnh
        transforms.RandomResizedCrop(224),  # Cắt ảnh ngẫu nhiên
        transforms.ToTensor(),  # chuyển ảnh sang tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa dữ liệu
    ])
    test_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # Lấy danh sách class
    class_names = train_dataset.classes
    class_indices = train_dataset.class_to_idx

    # In thông tin dataset
    # print("Classes:", class_names)
    # print("Class Indices:", class_indices)
    # print(f"Train samples: {len(train_dataset)}")
    # print(f"Validation samples: {len(val_dataset)}")
    # print(f"Test samples: {len(test_dataset)}")
    # image, label = train_dataset[0]  # Lấy mẫu đầu tiên
    # print(f"Image shape: {image.shape}")  # Kích thước ảnh
    # Image shape: torch.Size([3, 224, 224])
    '''
    MODEL 
    đầu tiên ta sử dụng thư viện skl trọng số để lấy trọng số
    Sử dụng class weight để tránh imbalanced data, bằng cách Khi một số lớp có quá nhiều ảnh so với lớp khác, 
    ta cần gán trọng số lớn hơn cho các lớp hiếm để tránh mô hình bị thiên vị.

    '''


    def compute_class_weights(train_path):
        files = [i.replace('\\', '/') for i in glob.glob(train_path + "//*//*")]
        labels = [os.path.dirname(i).split("/")[-1] for i in files]
        class_labels = sorted(set(labels))  # lấy danh sách duy nhất, sắp xếp theo bảng chữ cái
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array(class_labels), y=labels)
        '''
        class_weight='balanced':
        Tính toán trọng số sao cho tổng số mẫu của các lớp cân bằng nhau.
        classes=np.array(class_labels): Chuyển danh sách lớp thành mảng NumPy.
        y=labels: Đầu vào là danh sách nhãn ảnh.
        '''
        return dict(zip(class_labels, class_weights))


    class_weight_dict = compute_class_weights(train_dir)  # dùng khi khai báo hàm loss, vì crossEntropyloss yêu cầu 1 tensor nên ta cần chuyển đổi nó thành tensor
    # Chuyển đổi dict thành list
    class_weights_list = list(class_weight_dict.values())
    # Chuyển list thành tensor
    class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32)
    # print(class_weight_dict)

    '''
    +) Chúng ta sẽ tạo callback Dừng training nếu accuracy > ?% và val_accuracy > ?%: khi model chạm ngưỡng sẽ ngừng chạy 
    +) EarlyStopping: Nếu val_accuracy không cải thiện trong 5 epoch liên tiếp, training sẽ dừng.
    restore_best_weights=True: Khôi phục trọng số tốt nhất.
    +)ReduceLROnPlateau:Nếu val_accuracy không tăng trong 3 epoch, giảm learning rate đi 20%
    Giúp mô hình không bị "mắc kẹt" tại local minimum
    +) Mô hình mình lựa chọn sẽ là Xception được training bởi tập imagenet 
    '''


    class EarlyStopping:
        def __init__(self, patience=5, lr=0.001, target_acc=0.95):
            self.patience = patience
            self.lr = lr
            self.target_acc = target_acc
            self.best_acc = 0
            self.counter = 0

        def __call__(self, val_acc, train_acc):
            if train_acc > self.target_acc and val_acc > self.target_acc:
                print(f"\n Accuracy & Val Accuracy > {self.target_acc * 100}%, stopping training.")
                return True  # Dừng training
            elif val_acc - self.best_acc < self.lr:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"\n Early stopping after {self.patience} epochs.")
                    return True
            else:
                self.best_acc = val_acc
                self.counter = 0
            return False




    # khoi tao
    num_iter = len(train_loader) # số lượng iter chayj trong mỗi loader
    num_classes = len(class_names)  # class_names = danh sách tên các lớp
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    # Khởi tạo model
    model = XceptionModel(num_classes=num_classes)
    # Khởi tạo optimizer và loss function
    writer = SummaryWriter(args.logging) # dùng sumary để in ra cấu hình, tuy nhiên cần import thư viện vào
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(class_weights_tensor)
    # Khởi tạo checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0
    # quá trình train

    early_stopping = EarlyStopping(patience=5, target_acc=0.95)
    # Bắt đầu đo thời gian huấn luyện
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour="green")
        # running_loss = 0.0
        correct, total = 0, 0
        for iter, (inputs, labels) in enumerate(progress_bar):
            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iter, loss))
            writer.add_scalar("Train/Loss", loss, epoch*num_iter+iter)
            #backward
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()  # Đặt mô hình vào chế độ đánh giá
        all_val_predicteds = []
        all_val_labels = []
        for iter, (val_inputs, val_labels) in enumerate(val_loader): # val_loader là DataLoader của tập validation
            all_val_labels.extend(val_labels)
            with torch.no_grad():
                val_outputs = model(val_inputs)
                predicted = model(val_inputs)
                indices = torch.argmax(predicted, dim=1)
                all_val_predicteds.extend(indices)
                loss_val = criterion(predicted,val_labels)

        all_val_labels = [label.item() for label in all_val_labels]
        all_val_predicteds = [prediction.item() for prediction in all_val_predicteds]

        val_accuracy = accuracy_score(all_val_labels, all_val_predicteds)

        # Kiểm tra điều kiện dừng sớm
        if early_stopping(val_accuracy, train_acc):
            print('checking')
            break  # Thoát khỏi vòng lặp huấn luyện nếu điều kiện dừng sớm thỏa mãn
        plot_confusion_matrix(writer, confusion_matrix(all_val_labels, all_val_predicteds), class_names=class_names, epoch=epoch)


        print("Epoch {}: Accuracy: {}".format(epoch+1, val_accuracy))
        writer.add_scalar("Val/Accuracy", val_accuracy, epoch)
                # torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))
        checkpoint = {
            "epoch": epoch+1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
            }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))


    # Tính toán thời gian huấn luyện
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training time: {duration:.2f} seconds")
    print(f"Training time: {duration / 60:.2f} minutes")
    print(f"Training time: {duration / 3600:.2f} hours")
















