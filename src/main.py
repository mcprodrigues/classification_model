from flask import Flask, request, jsonify
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from flask_cors import CORS

from PIL import Image
import io

class_mapping = {
    0: 'cat',
    1: 'cow',
    2: 'goat',
    3: 'horse',
    4: 'iguana',
    5: 'lizard',
    6: 'ostrich',
    7: 'peacock',
    8: 'pigeon',
    9: 'possum'
}

# Modelo
backbone = models.resnet18(pretrained=True)

for name, param in backbone.named_parameters():
    if "layer4" in name or "fc" in name:
      param.requires_grad = True
    else:
      param.requires_grad = False

num_ftrs = backbone.fc.in_features
backbone.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.5),
    nn.Linear(512, 10)
)


# Carregando modelo
model = torch.load('./models/classification_model_v4', weights_only=False,  map_location=torch.device('cpu'))
model.eval()

# Threshold de confiança
confidence_threshold = 0.7

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "Hello, World!"

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']

    try:
        # Abrindo a imagem
        image = Image.open(file).convert('RGB')

        # Pré-processamento
        transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),   
        transforms.RandomHorizontalFlip(p=0.5),               
        transforms.RandomRotation(degrees=15),                
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        input_tensor = transform(image).unsqueeze(0)  # [1, 3, 512, 512]

        # Predição
        # Inferência com inference_mode (mais rápido e leve)
        with torch.inference_mode():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()

            if confidence < confidence_threshold:
                pred_class = 'uncertain'
            else:
                pred_class = class_mapping.get(pred_idx, 'Unknown')

        return jsonify({
            'prediction': pred_class,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    