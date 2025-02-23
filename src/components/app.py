from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from flask_cors import CORS
import torch.nn as nn
import torchvision.models as models

app = Flask(__name__)
CORS(app)  # Enable CORS

# Path to the trained model
best_model_path = "/home/aditya/Desktop/Xyaa/project_new/best_model.pth"

# Load Pretrained EfficientNet & Modify for 4 Classes
class BrainTumorEfficientNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorEfficientNet, self).__init__()
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorEfficientNet().to(device)

# Class labels
class_labels = ["glioma", "meningioma", "pituitary", "notumour"]

# Load best model
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

def predict_image(image):
    """Predicts the label for a given image object."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    image = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        output = model(image)  
        predicted_class = torch.argmax(output, dim=1).item()  

    predicted_label = class_labels[predicted_class]  
    print(f"Predicted Label: {predicted_label}")
    return predicted_label

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint to receive an image and return the prediction."""
    
    # **Check if a file was uploaded**
    if 'file' not in request.files:
        print("❌ No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        print("❌ Empty file received")
        return jsonify({'error': 'Empty filename'}), 400

    print(f"✅ Received file: {file.filename}")

    try:
        # **Read image from request**
        image = Image.open(io.BytesIO(file.read())).convert('RGB')  # Convert to RGB
        print("✅ Image loaded successfully")

        # **Run the model prediction**
        predicted_label = predict_image(image)

        # **Convert processed image to Base64**
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'diagnosis': predicted_label,
            'image': img_base64
        })

    except Exception as e:
        print(f"❌ Error processing image: {e}")  # Log error for debugging
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
