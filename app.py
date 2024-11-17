import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
import io

# Define class names
class_names = ['Aroncillo', 'D101', 'Davao Selection', 'Duyaya', 'Kob Basketball', 'Kob White', 'Lacson', 'Monthong', 'Native', 'Puyat', 'Unknown']

# Define your ResNeXt model class
class ResNeXtModel(nn.Module):
    def __init__(self):
        super(ResNeXtModel, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=False, num_classes=11)

    def forward(self, x):
        return self.model(x)

# Load the model
model_path = './best_model.pth'
model = ResNeXtModel()
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Define a minimal preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow external requests

# Define a route for the root URL
@app.route('/')
def index():
    return 'Welcome to the Image Classification API'

# Define a route for the favicon request
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content for favicon

# Define a route to handle image upload, processing, and model inference
@app.route('/image', methods=['POST'])
def load_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file part"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = transform(image)  # Apply the preprocessing transform
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
            predicted_class_name = class_names[predicted_class.item()]

        return jsonify({
            "predicted_class_name": predicted_class_name
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)