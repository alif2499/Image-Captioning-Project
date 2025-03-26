# Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from utility import  get_inference_model, generate_caption
import json
import argparse
from settings_inference import *

# Generate new caption from input image
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument('--image', help="Path to image file.")
parser.add_argument('--model', type=str, default='EfficientNetB0', help='Specify the pretrained model name (e.g., EfficientNetB0, EfficientNetB1, ResNet50, MobileNetV2)')
image_path = parser.parse_args().image
model_name = parser.parse_args().model

# Get tokenizer layer from disk
tokenizer = tf.keras.models.load_model(tokernizer_path)
tokenizer = tokenizer.layers[1]

# Get model
model = get_inference_model(get_model_config_path, model_name)

# Load model weights
model.load_weights(get_model_weights_path)

# Function to write JSON data to file
def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Loading the model config file
with open(get_model_config_path) as json_file:
    model_config = json.load(json_file)

# Path to test and prediction file
test_path = 'COCO_dataset\\test.json'
output_file1_path = 'prediction.json'

# Loading the test data
with open(test_path, 'r') as json_file:
    data = json.load(json_file)

# Initializing dictionary for saving predictions
prediction = {}

# Check if the --image_path argument is provided
if image_path is not None:
    # Perform testing with the provided argument value
    text_caption = generate_caption(image_path, model, tokenizer, model_config["SEQ_LENGTH"])
    print("PREDICT CAPTION : %s" %(text_caption))
else:
    # Perform default prediction on test data and store the predictions in prediction.json
    for key, value in data.items():
        text_caption = generate_caption(key, model, tokenizer, model_config["SEQ_LENGTH"])
        print("PREDICT CAPTION : %s" %(text_caption))
        prediction[key] = text_caption
        write_json_file(prediction, output_file1_path)
