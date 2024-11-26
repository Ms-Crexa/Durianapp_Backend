from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from rembg import remove
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the Image Classification API'

# Define class names
class_names = ['Aroncillo', 'D101', 'Davao Selection', 'Duyaya', 
               'Kob Basketball', 'Kob White', 'Lacson', 'Monthong', 
               'Native', 'Puyat', 'Unknown']

# Function to add a white background to an image
def add_white_background(image):
    if image.mode in ('RGBA', 'LA'):
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        white_bg.paste(image, mask=image.split()[-1])
        return white_bg
    return image

# Load the model
def load_model(model_path, device):
    model = models.resnext50_32x4d(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, len(class_names))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model_path = './final_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(model_path, device)

# Prediction function
def predict(image_bytes, model, device, threshold=0.6):
    try:
        # Load image and remove background
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        print("Original image loaded successfully.")
        
        image_bytes = remove(image_bytes)
        print("Background removed successfully.")
        
        bg_removed_image = Image.open(BytesIO(image_bytes)).convert('RGBA')

        # Add white background
        image_with_white_bg = add_white_background(bg_removed_image)

        # Transform image
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = data_transforms(image_with_white_bg).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get the predicted class and confidence
        predicted_idx = torch.argmax(probabilities).item()
        predicted_score = probabilities[predicted_idx].item()
        predicted_class = class_names[predicted_idx]

        if predicted_score < threshold:
            predicted_class = "Unknown"

        print(f"Predicted Class: {predicted_class}, Score: {predicted_score}")
        return predicted_class

    except Exception as error:
        print(f"Error in prediction: {str(e)}")
        return str(error)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['image']
        image_bytes = file.read()
        print("Image file received.")

        predicted_class = predict(image_bytes, model, device)

        if not predicted_class:
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify({"predicted_class": predicted_class}), 200

    except Exception as e:
        print(f"Error in API endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
