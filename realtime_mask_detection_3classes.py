# -*- coding: utf-8 -*-
"""
REAL-TIME FACE MASK DETECTION (3 CLASSES) - USING WEBCAM
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =============== CONFIG ===============
MODEL_PATH = "mask_detector_3classes.h5"  # Path to your trained model
PROTOTXT_PATH = "face_detector/deploy.prototxt"
CAFFE_MODEL_PATH = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CLASSES = ["without_mask", "incorrect_mask", "with_mask"]
COLORS = [(0, 0, 255), (255, 165, 0), (0, 255, 0)]  # Red, Orange, Green

# =============== LOAD MODELS ===============
print("[INFO] Loading models...")
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFE_MODEL_PATH)
mask_net = load_model(MODEL_PATH)

# =============== FUNCTION TO DETECT MASKS ===============
def detect_and_predict_mask(frame, face_net, mask_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Minimum confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding boxes are within frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract face ROI, preprocess for mask detection
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return (locs, preds)

# =============== START WEBCAM ===============
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))  # Resize for better display
    (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (without_mask, incorrect_mask, with_mask) = pred

        # Determine the class and color
        if with_mask > without_mask and with_mask > incorrect_mask:
            label = "No Mask"
            color = COLORS[2]  # Green
            confidence = with_mask
        elif incorrect_mask > without_mask and incorrect_mask > with_mask:
            label = "Mask"
            color = COLORS[1]  # Orange
            confidence = incorrect_mask
        else:
            label = "Mask (Incorrect)"
            color = COLORS[0]  # Red
            confidence = without_mask

        label = f"{label}: {confidence * 100:.2f}%"

        # Display bounding box and label
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Face Mask Detection (3 Classes)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()