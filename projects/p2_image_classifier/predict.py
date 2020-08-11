import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224,224))
    image = tf.cast(image, tf.float32)
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    # load the image and prepare the image
    image = Image.open(image_path)
    image = process_image(np.asarray(image))
    image = np.expand_dims(image, axis=0)
    # get the top k predictions
    predictions = model.predict(image)
    probs, labels = tf.math.top_k(predictions, top_k, sorted=False)
    return probs.numpy().squeeze(), (labels+1).numpy().squeeze().astype(str) 
    # labels+1 because the mapping starts at 1 while the labels starts with 0

# parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('image_path')
parser.add_argument('model')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names')
args = parser.parse_args()

# load model
model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})
# print(model.summary())

probs, classes = predict(args.image_path, model, args.top_k)
print(probs)
print(classes)

# check if mapping was provided, either print labels or class names when printing the probabilities
if args.category_names is not None:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        # print the resuls
        for k in range(classes.size):
            print("Class:", class_names[classes[k]], "Probability:", probs[k]) 

else:
    # print the resuls
    for k in range(classes.size):
        print("Class:", classes[k], "Probability:", probs[k]) 

        