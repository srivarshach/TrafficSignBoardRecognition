import os
import numpy as np
import cv2
from keras.models import load_model
from tkinter import Tk, filedialog, Label, Button, ttk
from PIL import Image, ImageTk

MODEL_PATH ='C://Users//mayan//OneDrive//Desktop//PROJECT//MINI PROJECT//traffic_sign_recognition//model.h5'

model = load_model(MODEL_PATH)

class TrafficSignRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Sign Recognition")
        master.geometry("600x400")
        master.configure(bg="#f0f0f0")

        self.label_trafficlights = Label(master)
        self.label_trafficlights.pack(pady=10)

        self.label_select = Label(master, text="Click below to select an image for prediction:", bg="#f0f0f0", fg="#333333", font=("Arial", 12))
        self.label_select.pack()

        self.select_button = Button(master, text="Select Image", command=self.select_image, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.select_button.pack()

        self.label_selected_image = Label(master)
        self.label_selected_image.pack(pady=10)

        self.result_label = Label(master, text="", bg="#f0f0f0", fg="#333333", font=("Arial", 12))
        self.result_label.pack(pady=10)

        self.quit_button = Button(master, text="Quit", command=master.quit, bg="#f44336", fg="white", font=("Arial", 12))
        self.quit_button.pack()

        self.load_trafficlights_image()

    def load_trafficlights_image(self):
        img_path = "C://Users//mayan//OneDrive//Desktop//PROJECT//MINI PROJECT//traffic_sign_recognition//image.jpg"  # Path to your traffic lights image
        img = Image.open(img_path)
        img = img.resize((200, 200), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.label_trafficlights.config(image=photo)
        self.label_trafficlights.image = photo

    def grayscale(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def equalize(self, img):
        img = cv2.equalizeHist(img)
        return img

    def preprocessing(self, img):
        img = self.grayscale(img)
        img = self.equalize(img)
        img = img / 255
        return img

    def getClassName(self, classNo):
        class_names = [
            'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
            'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
            'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
            'No passing', 'No passing for vehicles over 3.5 metric tons',
            'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
            'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
            'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
            'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
            'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
            'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
            'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
            'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
            'Keep left', 'Roundabout mandatory', 'End of no passing',
            'End of no passing by vehicles over 3.5 metric tons'
        ]
        return class_names[classNo]

    def model_predict(self, img):
        img = cv2.resize(img, (32, 32))
        img = self.preprocessing(img)
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        preds = self.getClassName(classIndex)
        return preds

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                preds = self.model_predict(img)
                self.result_label.config(text=f"Predicted Road Sign: {preds}")
                
                # Display selected image
                img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((200, 200), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.label_selected_image.config(image=photo)
                self.label_selected_image.image = photo
            else:
                self.result_label.config(text=f"Error reading image: {file_path}")
        else:
            self.result_label.config(text="No image selected.")

root = Tk()
app = TrafficSignRecognitionApp(root)
root.mainloop()
