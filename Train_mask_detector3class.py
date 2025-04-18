# -*- coding: utf-8 -*-
"""
FIXED VERSION - Face Mask Detection with 3 Classes
"""

import os
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

# Fix warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# =============== CONFIG ===============
INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
CLASSES = ["without_mask", "incorrect_mask", "with_mask"]
DATASET_PATH = "dataset"

# =============== LOAD DATASET ===============
def load_dataset():
    print("[INFO] Loading images...")
    data = []
    labels = []
    
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            
            try:
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = image.resize(IMAGE_SIZE)
                image = img_to_array(image)
                image = preprocess_input(image)
                
                data.append(image)
                labels.append(class_name)
            except Exception as e:
                print(f"[WARNING] Skipped {image_path}: {str(e)}")
                continue
    
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    print(f"[INFO] {len(data)} images loaded")
    return data, labels

# =============== PREPARE DATA ===============
data, labels = load_dataset()

le = LabelEncoder()
integer_labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(
    data, integer_labels,
    test_size=0.2,
    stratify=integer_labels,
    random_state=42
)

trainY = to_categorical(trainY, num_classes=len(CLASSES))
testY = to_categorical(testY, num_classes=len(CLASSES))

# =============== BUILD MODEL ===============
def build_model():
    # FIXED: Added missing parenthesis
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))  # <- This was missing
    )
    
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(len(CLASSES), activation="softmax")(head_model)
    
    model = Model(inputs=base_model.input, outputs=head_model)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

model = build_model()

# =============== TRAIN MODEL ===============
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"])

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS
)

# =============== EVALUATE ===============
predictions = model.predict(testX, batch_size=BATCH_SIZE)
predictions = np.argmax(predictions, axis=1)

print(classification_report(
    testY.argmax(axis=1),
    predictions,
    target_names=le.classes_))

# =============== SAVE MODEL ===============
model.save("mask_detector_3classes.h5")
print("[INFO] Model saved successfully!")

# =============== PLOT RESULTS ===============
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("training_plot.png")