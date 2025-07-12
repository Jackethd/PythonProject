import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os


model_path =r"C:\Users\jacket\PycharmProjects\PythonProject\ty_model.keras"
if not os.path.exists(model_path):
    print(f"Model dosyası bulunamadı: {model_path}")
    exit()

model = tf.keras.models.load_model(model_path)


img_path =r"C:\Users\jacket\PycharmProjects\PythonProject\dataset\train\not fractured\4-rotated2-rotated3-rotated1-rotated1.jpg"
if not os.path.exists(img_path):
    print(f"Görüntü dosyası bulunamadı: {img_path}")
    exit()


img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0


predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=-1)[0]


class_names = ["Kırık(Fractured)", "Sağlam (Healthy)"]
print(f"Tahmin Edilen Sınıf: {class_names[predicted_class]}")
confidence = predictions[0][predicted_class] * 100
print(f"Güven Skoru: {confidence:.2f}%")
