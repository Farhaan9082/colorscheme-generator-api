from flask import Flask, request, jsonify, json
from flask_cors import CORS
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
from numpy import asarray
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

@app.route("/color", methods=["POST"])
def generate_color():
    image = request.files['image'].read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def preprocess(raw):
        image = cv2.resize(raw, (900, 600), interpolation = cv2.INTER_AREA)                                          
        image = image.reshape(image.shape[0]*image.shape[1], 3)
        return image
    
    def rgb_to_hex(rgb_color):
        hex_color = "#"
        for i in rgb_color:
            hex_color += ("{:02x}".format(int(i)))
        return hex_color
    
    def analyze(img):
        clf = KMeans(n_clusters = 4)
        color_labels = clf.fit_predict(img)
        center_colors = clf.cluster_centers_
        counts = Counter(color_labels)
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
        return hex_colors
    
    modified_image = preprocess(image)
    colorscheme = analyze(modified_image)

    return jsonify({'colors': colorscheme})

if __name__ == "__main__":
    app.run(debug=True)