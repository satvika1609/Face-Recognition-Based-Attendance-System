import os
import dlib
import csv
import numpy as np
import logging
import cv2

path_images_from_camera = "data/data_faces_from_camera/"
path_features_csv = "data/features_all.csv"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')


def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)

    logging.info("Detecting face from image: %-30s", path_img)

    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        logging.warning("No face detected in: %s", path_img)
        face_descriptor = 0
    return face_descriptor


def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)

    for img_name in photos_list:
        path_img = os.path.join(path_face_personX, img_name)
        features = return_128d_features(path_img)
        if features != 0:
            features_list_personX.append(features)

    if features_list_personX:
        features_mean = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean = np.zeros(128, dtype=float)
    return features_mean


def run():
    logging.basicConfig(level=logging.INFO)

    person_list = sorted(os.listdir(path_images_from_camera))

    with open(path_features_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for person_folder in person_list:
            full_path = os.path.join(path_images_from_camera, person_folder)
            logging.info("Processing folder: %s", full_path)

            features_mean = return_features_mean_personX(full_path)

            if len(person_folder.split('_', 2)) == 2:
                person_name = person_folder
            else:
                person_name = person_folder.split('_', 2)[-1]

            row = [person_name] + list(features_mean)
            writer.writerow(row)
            logging.info("Features written for: %s", person_name)

    logging.info("All features saved to: %s", path_features_csv)


if __name__ == '__main__':
    run()
