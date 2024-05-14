import mediapipe as mp
import numpy as np
import argparse
import cv2
import os


class FaceRec:
    def __init__(self, dataset_name, video_path):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.name = dataset_name
        self.time_interval = 100  # jumlah sample
        self.video_path = video_path

        self.cap = cv2.VideoCapture(self.video_path)
        self.path = f"{os.path.join(os.getcwd(), 'dataset')}/{self.name}/"
        self.is_done = False

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.img_counter = sum(os.path.isfile(os.path.join(self.path, item)) for item in os.listdir(self.path))

    def read_video(self):
        while self.cap.isOpened():
            success, image = self.cap.read()

            if not success:
                # print("Ignoring empty camera frame.")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            self.process_frame(image, results)

    def process_frame(self, image, results):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                cx_min, cy_min = w, h
                cx_max, cy_max = 0, 0

                for landmark in face_landmarks.landmark:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cx_min = min(cx_min, cx)
                    cy_min = min(cy_min, cy)
                    cx_max = max(cx_max, cx)
                    cy_max = max(cy_max, cy)

                mask = np.zeros((h, w), dtype=np.uint8)
                face_points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
                convex_hull = cv2.convexHull(np.array(face_points))
                cv2.fillConvexPoly(mask, convex_hull, 255)
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                cropped_face = masked_image[cy_min:cy_max, cx_min:cx_max]
                cropped_face = cv2.resize(cropped_face, (250, 250))
                # cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)

        if self.time_interval > 0:
            print(self.time_interval)
            img_name = rf"{self.path}{self.img_counter}.jpg"
            cv2.imwrite(img_name, cropped_face)
            print("{} written!".format(img_name))
            self.img_counter += 1
            self.time_interval -= 1
        else:
            print("done!")
            self.cap.release()
            self.is_done = True

    def start(self):
        self.read_video()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-d", "--dataset", help="dataset")
#     parser.add_argument("-s", "--sources", help="source path")
#     args = parser.parse_args()
#
#     face_rec = FaceRec(args.dataset, args.sources)
#     face_rec.start()
