import os
from PIL import Image
import torch
from torchvision import datasets, transforms

def get_class_names(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return sorted(classes)

def process_path(file_path, class_names, img_size=(64, 64)):
    img = Image.open(file_path).convert('L')
    img = img.resize(img_size)
    img = transforms.ToTensor()(img)

    class_name = os.path.basename(os.path.dirname(file_path))
    label = class_names.index(class_name)
    return img, label

def load_dataset(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.classes