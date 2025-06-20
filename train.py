import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights,
    convnext_base,ConvNeXt_Base_Weights
)
import torch.nn as nn
from tqdm import tqdm

def get_model(name, num_classes, device):
    if name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'efficientnet':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'vit':
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name == 'convnext':
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")
    return model.to(device)

def train_model(data_dir='data', model_name='resnet50', num_classes=10, batch_size=32, lr=0.001, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transforms_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    dataset_train = ImageFolder(f'{data_dir}/train', transforms_train)
    dataset_test = ImageFolder(f'{data_dir}/test', transforms_test)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=4)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,num_workers=4)

    model = get_model(model_name, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def evaluate():
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in loader_test:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        count = 0
        loop = tqdm(loader_train, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1
            loop.set_postfix(loss=running_loss/count, lr=lr)

        acc = evaluate()
        print(f"Epoch {epoch+1}/{epochs} validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')

    print(f'Best accuracy: {best_acc:.4f}')

if __name__ == "__main__":
    print("choose model to train:")
    model_choose = input("1 for resnet50, 2 for efficientnet, 3 for vit, 4 for convnext: ")
    
    model_dict = {
        '1': 'resnet50',
        '2': 'efficientnet',
        '3': 'vit',
        '4': 'convnext'
    }

    if model_choose not in model_dict:
        print("invalid input, enter 1, 2, 3 or 4.")
    else:
        model_name = model_dict[model_choose]
        print(f"choose: {model_name} to train.")

        # modify these parameters as needed
        BATCH_SIZE = 32
        LR = 0.001
        EPOCHS = 5

        train_model(model_name=model_name, batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS)
        print("train done，model has been saved。")
