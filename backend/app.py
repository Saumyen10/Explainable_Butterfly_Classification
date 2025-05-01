from urllib import request
import torch #type: ignore
import torch.nn as nn           #type: ignore
import torch.nn.functional as F             #type: ignore
from torchvision import models, transforms              # type: ignore
from torchcam.methods import SmoothGradCAMpp            # type: ignore
from torchcam.utils import overlay_mask                     # type: ignore
from torchvision.transforms.functional import to_pil_image   # type: ignore
from PIL import Image       #type: ignore
import matplotlib.pyplot as plt         #type: ignore
import numpy as np # type: ignore           
import os
import uuid
from flask import Flask, request, jsonify, send_from_directory     # type: ignore
from flask_cors import CORS  # type: ignore

from clipcap_inference import generate_caption


app = Flask(__name__) # type: ignore
CORS(app)       #type:ignore

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = models.resnet34(pretrained=False)
num_classes = 51
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model/ResNet34_Indian_Species_Model.pth", map_location=device))
model.to(device)
model.eval()

#Butterfly classification based on species
species_classes = {
    0: 'Bamboo tree brown', # Correct place
    1: 'Banded ace', # Correct Place
    2: 'Banded tree brown',# Correct place
    3: 'Chestnut angle', # Correct place
    4: 'Common Evening Brown', #correct place
    5: 'Common jay', # Correct Place
    6: 'Common bluebottle', # Correct place
    7: 'Common cerulean', # Correct place
    8: 'Common duffer', # Correct Place
    9: 'Common gull', # Correct Place
    10: 'Common Peacock', #correct place
    11: 'Common red forester', # Correct Place
    12: 'Common spotted flat', # Correct place
    13: 'Common tree brown',  #correct place
    14: 'Common windmill', #correct place
    15: 'Dark evening brown', #correct place
    16: 'Dusky diadem',# Correct Place
    17: 'Elbowed pierrot', #correct place
    18: 'Fluffy tit',# Correct place ambiguous
    19: 'Fulvous pied flat', #correct place
    20: 'Grass Demon', #Correct Place
    21: 'Great orange tip', # Correct Place
    22: 'Indian skipper', # Correct Place
    23: 'Indian cabbage white', #correct place
    24: 'Jungle glory', # Correct Place
    25: 'Long banded silverline', # Correct place
    26: 'Mottled emigrant', #correct place
    27: 'Northern jungle queen', #correct place
    28: 'Orchid tit', # Correct Place
    29: 'Plain puffin', #correct place
    30: 'Psyche', # Correct place
    31: 'Punchinello', #correct place
    32: 'Purple sapphire', # Correct Place
    33: 'Redbase Jezebel', # Correct Place
    34: 'Rounded striped parrot', # Correct place
    35: 'Small green awlet', # Correct Place
    36: 'Spotted sawtooth',# Correct Place
    37: 'Tiger hopper',# Correct place
    38: 'Tree yellow', # Correct place
    39: 'Water snow flat', # Correct Place
    40: 'Wax dart', # Correct Place
    41: 'Yamfly',   # Correct Place
    42: 'Yellow orange tip',# Correct Place
    43: 'Chain swordtail', # Correct Place
    44: 'Common batwing', # Correct Place
    45: 'Common birdwing', # Correct Place
    46: 'Common raven', #correct place
    47: 'Common rose', # Correct Place
    48: 'Common tit', # Correct Place Ambiguous
    49: 'Five bar swordtail', #correct place
    50: 'Spotted royal' # Correct place
}

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Initialize CAM extractor
cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')

# Classification + heatmap logic
def predict_and_generate_cam(image_path, threshold=0.30):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)
    max_prob, predicted_class = torch.max(probabilities, 1)

    if max_prob.item() < threshold:
        label = "Unknown"
        heatmap_path = None
    else:
        class_idx = predicted_class.item()
        label = species_classes[class_idx]

        # Generate heatmap
        activation_map_list = cam_extractor(class_idx, outputs)
        activation_map = activation_map_list[0]

        heatmap = to_pil_image(activation_map.squeeze(0).cpu(), mode='F')
        result = overlay_mask(image, heatmap, alpha=0.5)

        heatmap_filename = f"heatmap_{uuid.uuid4()}.png"
        heatmap_path = os.path.join("static", heatmap_filename)
        result.save(heatmap_path)

    return label, max_prob.item(), heatmap_path


@app.route('/', methods=['GET'])
def home():
    return "🔥 Flask backend is running! Upload route is at /upload", 200

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

#upload image# Upload route
@app.route("/upload-image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the uploaded file temporarily
        filename = f"{uuid.uuid4()}.jpg"
        upload_folder = "static"
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Step 1: Predict the label and generate heatmap
        label, confidence, heatmap_path = predict_and_generate_cam(file_path, threshold=0.30)

        # Step 2: Generate textual description using ClipCap
        description = generate_caption(file_path)

        # print(f"Generated description: {description}")

        # Optional: Clean up the uploaded original image if you don't want to store it
        if os.path.exists(file_path):
            os.remove(file_path)

        # Step 3: Prepare URL to heatmap
        if heatmap_path:
            heatmap_url = f"/{heatmap_path.replace(os.sep, '/')}"
        else:
            heatmap_url = None

        # Step 4: Return full response
        return jsonify({
            "predicted_label": label,
            "confidence": confidence,
            "heatmap_url": heatmap_url,
            "description": description
        }), 200



