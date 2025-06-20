import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_name):
    if model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load('best_resnet50.pth', map_location=device))
    elif model_name == 'efficientnet':
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
        model.load_state_dict(torch.load('best_efficientnet.pth', map_location=device))
    else:
        raise ValueError("Invalid model name.")
    model.to(device)
    model.eval()
    return model

def get_animal_intro(animal_class):
    intro_path = f'animal_intro/{animal_class}.txt'
    if os.path.exists(intro_path):
        with open(intro_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return "no introduction available for this animal."

def predict_with_intro(image_pil, model_choice):
    model = load_model(model_choice)
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu()
        top5_probs, top5_idxs = torch.topk(probs, 5)

    result_dict = {class_names[i]: float(p) for i, p in zip(top5_idxs, top5_probs)}
    top1_class = class_names[top5_idxs[0].item()]
    intro = get_animal_intro(top1_class)

    return result_dict, intro


with gr.Blocks(title="animal recognition & introduce system",) as demo:
    gr.Markdown("## animal recognition & introduce system")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="upload image", interactive=True)
            model_choice = gr.Radio(choices=["resnet50", "efficientnet"], label="选择模型", value="resnet50")
            predict_btn = gr.Button("start prediction")
        with gr.Column(scale=1):
            confidence_output = gr.Label(label="confidence coefficient（Top-5）")
            intro_output = gr.Textbox(label="animal introduction", lines=5)

    predict_btn.click(
        fn=predict_with_intro,
        inputs=[image_input, model_choice],
        outputs=[confidence_output, intro_output]
    )
if __name__ == "__main__":
    demo.launch()
