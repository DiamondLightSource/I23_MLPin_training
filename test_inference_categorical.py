import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
from datetime import datetime

model = tf.keras.models.load_model("save_batch16.h5")
model.summary()

test_data_dir = "test_13022025"
categories = ["dark", "light", "pinoff", "pinon"]
img_height = 800
img_width = 800

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    #img = img / 255.0  
    return img

def run_inference():
    correct_predictions = 0
    total_predictions = 0

    for category in categories:
        category_dir = os.path.join(test_data_dir, category)
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)
            img = preprocess_image(image_path)
            img = np.expand_dims(img, axis=0) 

            prediction = model.predict(img)
            predicted_label = categories[np.argmax(prediction)]
            print(f"Image: {image_name}, Prediction: {prediction}, Predicted Label: {predicted_label}, Actual Label: {category}")


            if predicted_label == category:
                correct_predictions += 1
            else:
                print(f"Misclassified: {image_name}, Predicted: {predicted_label}, Actual: {category}")
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    run_inference()