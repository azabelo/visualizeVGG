import torchvision.models as models
import urllib.request
import torchsummary
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.datasets import Caltech101
from PIL import Image
import os
import operator
import cv2
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


dataset = Caltech101(root = "./", download=True)
X, _ = create_training_data('./Caltech101')



vgg16 = models.vgg16(pretrained=True)

# Define the hook function
def get_layer_output_hook(module, input, output):
    layer_outputs.append(output)

# Register the hook
layer_outputs = []
layer_to_hook = 2 # The index of the layer
target_layer = vgg16.features[layer_to_hook]
target_layer.register_forward_hook(get_layer_output_hook)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image and apply transformations
imageIndex = 8000
image = Image.fromarray(X[imageIndex])
image_tensor = transform(image)

# Add batch dimension to image tensor
image_tensor = torch.unsqueeze(image_tensor, 0)

# Pass image through VGG16 model and get predictions
output = vgg16(image_tensor)
predictions = torch.nn.functional.softmax(output, dim=1)

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


featureMaps = layer_outputs[0]
# Print the output tensor
print(featureMaps.shape)

print(torchsummary.summary(vgg16, (3, 144, 144)))

#imgplot = plt.imshow(image)
#plt.imshow(featureMaps.detach().numpy()[0, featureIndex, :, :], cmap='gray')
#plt.show()


fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(16, 16))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
# Flatten the grid of subplots into a 1D array
axs = axs.flatten()


# Loop through the images and plot each one in a subplot
for i in range(64):
    # Get the ith image from the array
    img = featureMaps.detach().numpy()[0, i, :, :]

    # Plot the image in the ith subplot
    axs[i].imshow(img, cmap='hot')

    # Remove the axis ticks and labels from the subplot
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')

# Display the plot
plt.show()






