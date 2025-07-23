import tensorflow as tf
import os
from tensorflow import keras


model = tf.keras.models.load_model("20250224-154633_binary_batch32.h5")
model.summary()
classes = ["pinoff", "pinon"]
test_data_dir = "test_19022025_binary"

def get_first_image_size(data_dir):
    pinon_dir = os.path.join(data_dir, "pinon")
    files = [
        f for f in os.listdir(pinon_dir) if os.path.isfile(os.path.join(pinon_dir, f))
    ]
    if not files:
        raise FileNotFoundError("No files found in pinon directory")
    first_file = os.path.join(pinon_dir, files[0])
    img_bytes = tf.io.read_file(first_file)
    img = tf.image.decode_image(img_bytes)
    height = img.shape[0]
    width = img.shape[1]
    return height, width

def infer(image, class_):
    h, w = get_first_image_size(test_data_dir)
    img_in = keras.preprocessing.image.load_img(image, target_size=(int(h), int(w)))
    img_array = keras.preprocessing.image.img_to_array(img_in)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    if score < 0.05:
        print(f"{score} is {classes[0]}")
        predicted_label = classes[0]
    elif score > 0.95:
        print(f"{score} is {classes[1]}")
        predicted_label = classes[1]
    else:
        print(f"{score} is unknown")
        predicted_label = "unknown"

    if predicted_label == class_:
        print(f"Correctly classified {image} as {predicted_label}")
        return 1
    else:
        print(f"Incorrectly classified {image} as {predicted_label}")
        return 0


if __name__ == "__main__":
    correct = 0
    total = 0
    for class_ in classes:
        class_dir = os.path.join(test_data_dir, class_)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(test_data_dir, class_, image_name)
            correct += infer(image_path, class_)
            total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
