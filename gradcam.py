# pytorch autograd package
# last conv layer index

#in the future try to modularize the code from the very start

import torch
from torch import nn
import torchvision.models as models
import urllib.request
from torchsummary import summary
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.datasets import Caltech101
from PIL import Image
import os
import operator
import cv2
import matplotlib.cm as cm
import numpy as np


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

"""
dataset = Caltech101(root = "./", download=True)
X, _ = create_training_data('./Caltech101')
"""

# Load the VGG16 model from torchvision
vgg16 = models.vgg16(pretrained=True)

# Define the indices for splitting the model
lastConvLayerIndex = 31


# Split the model into two parts based on the defined index
vgg_LastConv = torch.nn.Sequential(*list(vgg16.features.children())[:lastConvLayerIndex])
vgg_Classifier = torch.nn.Sequential(*list(vgg16.features.children())[lastConvLayerIndex:], vgg16.avgpool, nn.Flatten(), *list(vgg16.classifier.children()))


# Print the summary of both the models
print("Model Part 1:")
print(vgg_LastConv)
summary(vgg_LastConv, (3, 224, 224))
print("\nModel Part 2:")
print(vgg_Classifier)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#image from Caltech101
#imageIndex = 6500
#image = Image.fromarray(X[imageIndex])

# or use url:

#bus next to car
url = "https://www.gannett-cdn.com/authoring/2019/12/06/NRCD/ghows-SR-99121660-4c3b-12cb-e053-0100007f872c-63f1b23a.jpeg?width=660&height=471&fit=crop&format=pjpg&auto=webp"
#two cars
url = "https://i2-prod.gazettelive.co.uk/incoming/article15379246.ece/ALTERNATES/s615/1_Aston_Martin_Vantage__DB11_Volante-1.jpg"
# elephant
url = "https://i.imgur.com/Bvro0YD.png"
#url = "https://hips.hearstapps.com/hmg-prod/images/p90475606-highres-rolls-royce-phantom-1677268219.jpg?crop=0.663xw:0.496xh;0.136xw,0.372xh&resize=1200:*"
with urllib.request.urlopen(url) as url:
    image = Image.open(url)

image = image.convert('RGB')
image_tensor = transform(image)
# Add batch dimension to image tensor
image_tensor = torch.unsqueeze(image_tensor, 0)



convOutput = vgg_LastConv(image_tensor)
print(convOutput.shape)

"""last = Image.fromarray(convOutput[0,0,:,:].detach().numpy())
imgplot = plt.imshow(last)
plt.show()"""

classifierOutput = vgg_Classifier(convOutput)

predictions = torch.nn.functional.softmax(classifierOutput, dim=1)
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
class_names = urllib.request.urlopen(url).read().decode('utf-8').split('\n')

# Print top 5 predicted classes
num_classes = 1000  # VGG16 has 1000 output classes
top_predictions = torch.topk(predictions, k=5)
for i in range(5):
    class_idx = top_predictions.indices[0][i]
    class_prob = top_predictions.values[0][i]
    class_name = class_names[class_idx]
    print(f'Class name: {class_name}, Class probability: {class_prob.item()}')

#get specific class
class_idx = class_names.index('"sports car",')
#based on rankings
nth_Best = 0
class_idx = top_predictions.indices[0][nth_Best]


grads = torch.autograd.grad(outputs=classifierOutput[:, class_idx], inputs=convOutput, grad_outputs=torch.ones_like(classifierOutput[:, class_idx]), create_graph=True)
grads = grads[0].detach()



averagedGrads = torch.mean(grads, axis=(0, 2, 3))

convOutput = convOutput.detach()[0]

#not sure when the best time to use detach is???
product = convOutput * averagedGrads[:,None,None]

overlay = torch.sum(product, dim=0).squeeze().detach()

# not sure why its inverted, but that is why i subtracted it from 255
normalizedOverlay = np.uint8(255 * (overlay - overlay.min()) / (overlay.max() - overlay.min()))
print(normalizedOverlay)

plt.matshow(normalizedOverlay)
plt.show()

heatmap = cm.get_cmap("jet")(np.arange(256))[:, :3][normalizedOverlay]
heatmapImage = np.array(Image.fromarray(np.uint8(255*heatmap)).resize((np.array(image).shape[1],np.array(image).shape[0])))
plt.matshow(heatmapImage)
plt.show()


opacity = 0.4
layered = Image.fromarray((np.array(image) + opacity * heatmapImage).astype('uint8'), 'RGB')

plt.imshow(layered)
plt.show()
