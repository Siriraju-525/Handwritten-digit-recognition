import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data to fit the model
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('digit_recognition_model.h5')

# Load the trained model
model = load_model('digit_recognition_model.h5')

# Create the drawing interface
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Digit Recognizer')
        self.canvas = tk.Canvas(self, width=400, height=400, bg='white')
        self.canvas.pack()
        self.button_recognize = tk.Button(self, text='Recognize', command=self.recognize)
        self.button_clear = tk.Button(self, text='Clear', command=self.clear)
        self.button_recognize.pack()
        self.button_clear.pack()

        self.canvas.bind('<B1-Motion>', self.paint)
        self.image = Image.new('L', (400, 400), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (400, 400), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def recognize(self):
        # Preprocess the image
        img = self.image.resize((28, 28)).convert('L')
        img = np.array(img)
        img = 255 - img
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict the digit
        result = model.predict(img)
        digit = np.argmax(result)
        accuracy = np.max(result)
        self.title(f'Digit Recognizer - Prediction: {digit} - Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    app = App()
    app.mainloop()
