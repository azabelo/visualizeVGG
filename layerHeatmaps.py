import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
from matplotlib import pyplot as plt
from torch.nn import Module
import torchsummary
import numpy


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
image = Image.open('Caltech101/101_ObjectCategories/airplanes/image_0548.jpg')
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






