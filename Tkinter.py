import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os


model_path = r"C:\Users\jacket\PycharmProjects\PythonProject\ty_model.keras"
if not os.path.exists(model_path):
    print(f"Model dosyası bulunamadı: {model_path}")
    exit()

model = tf.keras.models.load_model(model_path)


class_names = ["Kırık (Fractured)", "Sağlam (Healthy)"]
img_size = (224, 224)


def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    confidence = predictions[0][predicted_class] * 100
    return class_names[predicted_class], confidence


def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("JPEG Files", "*.jpg"), ("PNG Files", "*.png"), ("All Files", "*.*")]
    )
    if file_path:

        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk


        prediction, confidence = predict_image(file_path)
        result_label.config(text=f"Tahmin: {prediction}\nGüven: {confidence:.2f}%")


root = tk.Tk()
root.title("X-Ray Kırık Tespit Uygulaması")
root.geometry("400x400")


title_label = Label(root, text="X-Ray Kırık Tespit Uygulaması", font=("Arial", 16))
title_label.pack(pady=10)


image_label = Label(root, text="Görüntü yükleyin", width=25, height=10, bg="gray")
image_label.pack(pady=10)


result_label = Label(root, text="Sonuç bekleniyor...", font=("Arial", 12))
result_label.pack(pady=10)


upload_button = Button(root, text="Görüntü Yükle", command=load_image)
upload_button.pack(pady=10)


root.mainloop()
