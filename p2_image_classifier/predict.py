import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import os
import logging
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

parser = argparse.ArgumentParser(description='Flower classification')
parser.add_argument('image_path', action='store', type=str)
parser.add_argument('model_path', action='store', type=str)
parser.add_argument('--top_k', action='store', dest='top_k', default=1, type=int)
parser.add_argument('--category_names', action='store', dest='category_path', type=str)
results = parser.parse_args()

if not os.path.isfile(results.image_path):
    raise Exception("Image path not found")

if not os.path.isfile(results.model_path):
    raise Exception("Model path not found")

if results.category_path and not os.path.isfile(results.category_path):
    print("Category path not found. Displaying numerical categories...")

if results.category_path:
    with open(results.category_path, 'r') as f:
        class_names = json.load(f)

IMG_SIZE = 224
def process_image(img):
    image = tf.cast(img, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image /= 255
    return image.numpy()

def to_classname(class_id):
    return class_names[str(class_id)]

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

def predict(image_path, model, top_k):
    img = np.asarray(Image.open(image_path))
    img = process_image(img)
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    
    top_predictions = np.argsort(predictions[0])[::-1][:top_k]
    
    return (predictions[0][top_predictions], top_predictions + 1)


model = load_model(results.model_path)
probabilities, class_ids = predict(results.image_path, model, results.top_k)

if results.category_path:
    class_ids = list(map(to_classname, class_ids))

for i in range(len(probabilities)):
    print(f'{class_ids[i]}: {"{:.2%}".format(probabilities[i])}')



