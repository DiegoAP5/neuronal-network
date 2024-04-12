from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)
model = load_model('modelo.h5')
classes = ['aurrera','caprice','coty','equate','garnier','head','herbal','nivea','pantene','vainilla']

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    image = np.array(image).reshape(-1, 64, 64, 1)
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]
    print('predicted class:', predicted_class)
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
