from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors
from datetime import datetime
import face_recognition
import mediapipe as mp
import numpy as np
import argparse
import os.path
import pickle
import math
import cv2
import os

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "JPG"}
mp_face_mesh = mp.solutions.face_mesh


class Main:
    def __init__(self, model, source):
        self.model = model  # model path
        self.source = source  # video source
        self.absent = False
        self.predictions = []
        self.accuracy = float

    @staticmethod
    def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo="ball_tree", verbose=False):
        X = []
        y = []

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        pass
                        # print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(
                        face_recognition.face_encodings(
                            image, known_face_locations=face_bounding_boxes
                        )[0]
                    )
                    y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                pass
                # print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm=knn_algo, weights="distance"
        )
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, "wb") as f:
                pickle.dump(knn_clf, f)

        return knn_clf

    def predict(self, X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
        accuracy = 0

        if knn_clf is None and model_path is None:
            raise Exception(
                "Must supply knn classifier either thourgh knn_clf or model_path"
            )

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, "rb") as f:
                knn_clf = pickle.load(f)

        X_face_locations = face_recognition.face_locations(X_frame)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(
            X_frame, known_face_locations=X_face_locations
        )

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [
            closest_distances[0][i][0] <= distance_threshold
            for i in range(len(X_face_locations))
        ]

        if are_matches:
            for i in range(len(X_face_locations)):
                distance = np.round(closest_distances[0][i][0] * 100)
                accuracy = 100 - np.round(distance)
                print(closest_distances[0][i][0])
                print(f"Accuracy Level: {accuracy}%")
                if accuracy < 55:
                    print("unknown")
                else:
                    print("known")

        # Predict classes and remove classifications that aren't within the threshold
        self.predictions = [
            (pred, loc) if rec else ("unknown", loc)
            for pred, loc, rec in zip(
                knn_clf.predict(faces_encodings), X_face_locations, are_matches
            )
        ]
        self.accuracy = accuracy

    @staticmethod
    def check_user_attendance(name):
        with open("absensi.csv", "r+") as f:  # file csv untuk menyimpan data di excel
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(",")
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f"\n{name},{dtString}")
                print("data berhasil ditambahkan!")
            else:
                print("data sudah ada di database")

    def process_predictions(self, frame, predictions, acc):
        for name, _ in predictions:
            name = name.encode("UTF-8")
            if acc > 75:  # treshold,sesuaikan dengan rata rata accuracynya
                self.check_user_attendance(name.decode())
                print("data saved!")
            else:
                print("Akurasinya kurang")

    def training(self):
        print("Training KNN classifier...")
        classifier = self.train(
            os.path.join(os.getcwd(), "dataset"),
            model_save_path=self.model,
            n_neighbors=3,
        )
        print("Training complete!")

    def run(self, frame):
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        try:
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = img.shape
                    cx_min, cy_min = w, h
                    cx_max, cy_max = 0, 0

                    for landmark in face_landmarks.landmark:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cx_min = min(cx_min, cx)
                        cy_min = min(cy_min, cy)
                        cx_max = max(cx_max, cx)
                        cy_max = max(cy_max, cy)

                    mask = np.zeros((h, w), dtype=np.uint8)

                    face_points = [
                        (int(landmark.x * w), int(landmark.y * h))
                        for landmark in face_landmarks.landmark
                    ]

                    convex_hull = cv2.convexHull(np.array(face_points))
                    cv2.fillConvexPoly(mask, convex_hull, 255)

                    masked_image = cv2.bitwise_and(img, img, mask=mask)
                    cropped_face = masked_image[cy_min:cy_max, cx_min:cx_max]
                    cropped_face = cv2.resize(cropped_face, (250, 250))

                    self.predict(cropped_face, model_path=self.model)
                    self.process_predictions(img, self.predictions, self.accuracy)
        except Exception as e:
            print(e)

        # Release resources
        face_mesh.close()
        cv2.destroyAllWindows()

# parser = argparse.ArgumentParser()
# parser.add_argument("mode", type=str)
# parser.add_argument("-m", "--model", help="model path")
# parser.add_argument(
#     "-s", "--sources", type=lambda x: int(x) if x.isdigit() else x, help="source path"
# )
# args = parser.parse_args()
# Training = Main(model=args.model, source=args.sources)
#
# if args.mode == "t":
#     Training.training()
# elif args.mode == "r":
#     Training.run()
