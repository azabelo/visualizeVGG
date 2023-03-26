import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import io
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from matplotlib import pyplot as plt
from  matplotlib import image
from torchvision.datasets import Caltech101
from torchvision.datasets import VisionDataset, ImageFolder
from PIL import Image
import numpy as np
import cv2
import operator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import resnet18, resnet50



def get_images(object_category, data_directory):
    if (not os.path.exists(data_directory)):
        print("Data directory not found. Are you sure you downloaded and extracted dataset properly?")
        return
    obj_category_dir = os.path.join(os.path.join(data_directory,"101_ObjectCategories"),object_category)
    images = [os.path.join(obj_category_dir,img) for img in os.listdir(obj_category_dir)]
    return images

def read_image(image_path):
    """Read and resize individual images - Caltech 101 avg size of image is 300x200, so we resize accordingly"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
    return img

def return_images_per_category(data_directory):
    categories = os.listdir(data_directory+"/101_ObjectCategories/")
    object_images_count_dict = {}
    for category in categories:
        object_images_count_dict[category] = len(os.listdir(data_directory+"/101_ObjectCategories/"+category))
    object_images_count_dict = sorted(object_images_count_dict.items(), key=operator.itemgetter(1), reverse=True)
    return object_images_count_dict

def create_training_data(data_directory):
    i = 0
    X = np.ndarray((8677, 224, 224, 3), dtype=np.uint8)
    Y = []
    print("Preparing X and Y for dataset...")
    for category,_ in return_images_per_category(data_directory):
        if category == 'BACKGROUND_Google':
            continue
        print("Processing images of ",category)
        for image in get_images(category, data_directory):
            if not image.endswith('.jpg'):
                # to escape hidden ipynb checkpoints and other unnecessary files
                continue
            resized = Image.fromarray(read_image(image)).resize((224, 224))
            X[i] = read_image(image)
            Y.insert(i,category)
            i += 1
        print("Images processed : ",i+1," of 8678")
    print("Datasets constructed")
    return X,Y

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(75 * 50 * 32, 512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 101)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 75 * 50 * 32)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == '__main__':
    mean, stdev = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # Define transforms for training phase
    train_transform = transforms.Compose([transforms.Resize(256),  # Resizes short size of the PIL image to 256
                                          transforms.CenterCrop(224),  # Crops a central square patch of the image
                                          # 224 because torchvision's AlexNet needs a 224x224 input!
                                          # Remember this when applying different transformations, otherwise you get an error
                                          transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
                                          transforms.Normalize(mean, stdev)
                                          # Normalizes tensor with mean and standard deviation
                                          ])

    # Define transforms for the evaluation phase
    eval_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, stdev)
                                         ])



    print("here1")
    dataset = Caltech101(root = "./", download=True)
    dolphin_imgs = get_images('dolphin', './Caltech101')
    print(dolphin_imgs[0])
    dolphin_test_img = read_image(dolphin_imgs[0])
    print(dolphin_test_img.shape)
    print(return_images_per_category('./Caltech101'))

    X, Y = create_training_data('./Caltech101')

    label_encoder = LabelEncoder()
    Y_integer_encoded = label_encoder.fit_transform(Y)

    Y_2d = np.array(Y).reshape(-1,1)
    encoder = OneHotEncoder()
    Y_one_hot = encoder.fit_transform(Y_2d)


    print("deleting old X")
    X_normalized = X.astype(np.float64) / 255
    del X

    print("splitting data")
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_normalized, Y_one_hot, test_size=0.25,
                                                                    random_state=42)

    Y_train = Y_train.toarray()
    X_train = X_train.transpose((0, 3, 1, 2))
    print(type(X_train))
    print(type(Y_train))
    train_dataset = MyDataset(X_train, Y_train)

## our data is now ready #####################



    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    vgg = vgg16(pretrained=True).classifier

    print(vgg(torch.tensor(X_train, dtype=torch.float32)))

    """model = ImageClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    print("here2")


  imgplot = plt.imshow(X_train[6001])
    plt.show()
    print(label_encoder.inverse_transform(np.array([np.argmax(Y_train[6001])])))


    print(Y_one_hot[0])
    print(np.argmax(Y_one_hot[0]))
    print(label_encoder.inverse_transform(np.array([np.argmax(Y_one_hot[0])])))


    imageRand = dataset.__getitem__(870)
    print(imageRand)
    print(os.listdir("./Caltech101/101_ObjectCategories"))

    imgplot = plt.imshow(imageRand[0])
    plt.show()
"""


