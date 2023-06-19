import pandas as pd

import cv2
import numpy as np
import dlib

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('fairface_label_train.csv')

# Load the facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

count=0
new_df = pd.DataFrame(columns=['eye_distance', 'eye_height', 'nose_height', 'face_width', 'jaw_angle', 'lip_width','eyebrow_gap', 'path'])
for index, row in df.iterrows():
  
    img = cv2.imread(row['file'])
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the imag
        faces = detector(gray)
        if faces is not None:
            
            for face in faces:
                # Get the facial landmarks for the face
                landmarks = predictor(gray, face)
                landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

                # Compute the eye distance
                eye_distance=np.linalg.norm(landmarks[36] - landmarks[45])
                
                                # Get the coordinates of the left eye
                left_eye_coords = landmarks[36:42]

                # Get the coordinates of the right eye
                right_eye_coords = landmarks[42:48]

                # Calculate the height of the left eye
                left_eye_height = (left_eye_coords[3][1] - left_eye_coords[1][1] + left_eye_coords[4][1] - left_eye_coords[2][1]) / 2

                # Calculate the height of the right eye
                right_eye_height = (right_eye_coords[3][1] - right_eye_coords[1][1] + right_eye_coords[4][1] - right_eye_coords[2][1]) / 2

                # Calculate the average height of the eyes
                eye_height = (left_eye_height + right_eye_height) / 2

                nose_height = np.linalg.norm(landmarks[30] - landmarks[27])
                lip_width = np.linalg.norm(landmarks[48] - landmarks[54])
                
                
                face_width =np.linalg.norm(landmarks[0] - landmarks[16])

                #jaw angle
                vec_AB = landmarks[6] - landmarks[8]
                vec_BC = landmarks[10] - landmarks[8]

                dot_product = np.dot(vec_AB, vec_BC)

                magnitude_AB = np.linalg.norm(vec_AB)
                magnitude_BC = np.linalg.norm(vec_BC)

                angle_radians = np.arccos(dot_product / (magnitude_AB * magnitude_BC))

                jaw_angle=np.degrees(angle_radians)

                eyebrow_gap = np.linalg.norm(landmarks[22] - landmarks[21])
                
                filename = row['file']
                
                # Add the values to the new DataFrame
                new_df = new_df.append({'file': filename, 'eye_distance': eye_distance, 'eye_height': eye_height,\
                                        'nose_height': nose_height, 'face_width': face_width,\
                                        'jaw_angle': jaw_angle, 'lip_width': lip_width, 'eyebrow_gap':eyebrow_gap},\
                                       ignore_index=True)
               
        else:
            pass

new_df.to_csv('gender_features_train.csv', index=False)

df1 = pd.read_csv('fairface_label_val.csv')

count=0
new_df = pd.DataFrame(columns=['file','eye_distance', 'eye_height', 'nose_height', 'face_width', 'jaw_angle', 'lip_width','eyebrow_gap', 'gender'])
for index, row in df1.iterrows():
    img = cv2.imread(row['file'])
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the imag
        faces = detector(gray)
        if faces is not None:
            count+=1
            for face in faces:
                # Get the facial landmarks for the face
                landmarks = predictor(gray, face)
                landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

                # Compute the eye distance
                eye_distance=np.linalg.norm(landmarks[36] - landmarks[45])
                
                                # Get the coordinates of the left eye
                left_eye_coords = landmarks[36:42]

                # Get the coordinates of the right eye
                right_eye_coords = landmarks[42:48]

                # Calculate the height of the left eye
                left_eye_height = (left_eye_coords[3][1] - left_eye_coords[1][1] + left_eye_coords[4][1] - left_eye_coords[2][1]) / 2

                # Calculate the height of the right eye
                right_eye_height = (right_eye_coords[3][1] - right_eye_coords[1][1] + right_eye_coords[4][1] - right_eye_coords[2][1]) / 2

                # Calculate the average height of the eyes
                eye_height = (left_eye_height + right_eye_height) / 2

                nose_height = np.linalg.norm(landmarks[30] - landmarks[27])
                lip_width = np.linalg.norm(landmarks[48] - landmarks[54])
                
                
                face_width =np.linalg.norm(landmarks[0] - landmarks[16])

                #jaw angle
                vec_AB = landmarks[6] - landmarks[8]
                vec_BC = landmarks[10] - landmarks[8]

                dot_product = np.dot(vec_AB, vec_BC)

                magnitude_AB = np.linalg.norm(vec_AB)
                magnitude_BC = np.linalg.norm(vec_BC)

                angle_radians = np.arccos(dot_product / (magnitude_AB * magnitude_BC))

                jaw_angle=np.degrees(angle_radians)

                eyebrow_gap = np.linalg.norm(landmarks[22] - landmarks[21])
                
                gender = row['gender']
                filename = row['file']
                
                # Add the values to the new DataFrame
                new_df = new_df.append({'file': filename, 'eye_distance': eye_distance, 'eye_height': eye_height,\
                                        'nose_height': nose_height, 'face_width': face_width,\
                                        'jaw_angle': jaw_angle, 'lip_width': lip_width, 'eyebrow_gap':eyebrow_gap, 'gender': gender},\
                                       ignore_index=True)
               
        else:
            pass

new_df.to_csv('gender_features_val.csv', index=False)