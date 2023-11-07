#!/usr/bin/env python
import csv
import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


def preprocess_landmarks(landmark_list):
    temp_list = np.array(landmark_list, dtype=np.float32)
    # Make coordinates relative to the first keypoint (wrist)
    temp_list = temp_list - temp_list[0]
    # Flatten the list
    temp_list = temp_list.flatten()
    # Normalize the points by the maximum absolute value to bring values between -1 and 1
    max_value = max(np.max(np.abs(temp_list)), 1)  # Avoid division by zero
    temp_list = temp_list / max_value
    return temp_list.tolist()

# Read the image
image = cv2.imread("C:\\Users\\joshu\\OneDrive\\Desktop\\mediapipe fun\\hand-gesture-recognition-mediapipe\\test.png")
if image is None:
    print("Error loading image")
else:
    print("Image loaded successfully")

# Process the image
results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# If hand landmarks are detected, write them to the CSV file
if results.multi_hand_landmarks:
    landmarks = results.multi_hand_landmarks[0]  # Assume only one hand is in the picture for simplicity
    landmark_list = calc_landmark_list(image, landmarks)
    preprocessed_list = preprocess_landmarks(landmark_list)

    with open('model/keypoint_classifier/keypoint.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([6969, *preprocessed_list])  # Replace '1' with the appropriate label for your image

    print('Processed landmarks for test.png and updated CSV')
else:
    print('No hand landmarks detected in test.png')

# Close the hand model session
hands.close()

print('Done!')

