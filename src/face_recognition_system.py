import cv2
import os
import numpy as np
import pickle


class FaceRecognitionSystem:
    def __init__(self, database_path="students_database"):

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.database_path = database_path
        self.labels_info = {}

        if not os.path.exists(database_path):
            os.makedirs(database_path)

    def train_recognizer(self):
        print("얼굴 인식기 훈련 시작...")

        faces = []
        labels = []
        label_id = 0
        self.labels_info = {}

        for student_id in os.listdir(self.database_path):
            student_dir = os.path.join(self.database_path, student_id)

            if os.path.isdir(student_dir):
                # 현재 학생 ID에 label_id 할당
                self.labels_info[label_id] = student_id

                for img_file in os.listdir(student_dir):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        img_path = os.path.join(student_dir, img_file)

                        try:
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                print(f"이미지를 로드할 수 없습니다: {img_path}")
                                continue

                            faces.append(img)
                            labels.append(label_id)
                        except Exception as e:
                            print(f"이미지 처리 중 오류 발생: {e}")

                label_id += 1

        if len(faces) > 0:
            self.face_recognizer.train(faces, np.array(labels))
            print(f"{len(faces)}개의 얼굴 이미지로 인식기 훈련 완료")

            model_path = os.path.join(self.database_path, "trained_model.yml")
            self.face_recognizer.write(model_path)

            labels_path = os.path.join(self.database_path, "labels_info.pkl")
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels_info, f)

            return True
        else:
            print("훈련할 얼굴 이미지가 없습니다.")
            return False

    def load_trained_model(self):
        model_path = os.path.join(self.database_path, "trained_model.yml")
        labels_path = os.path.join(self.database_path, "labels_info.pkl")

        if os.path.exists(model_path) and os.path.exists(labels_path):
            self.face_recognizer.read(model_path)

            with open(labels_path, 'rb') as f:
                self.labels_info = pickle.load(f)

            print("저장된 얼굴 인식 모델 로드 완료")
            return True

        print("저장된 모델이 없습니다. 새로 훈련이 필요합니다.")
        return False

    def detect_faces(self, img):
        if img is None:
            return []

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        return faces

    def recognize_face(self, face_img):
        try:
            label, confidence = self.face_recognizer.predict(face_img)

            if label in self.labels_info:
                return self.labels_info[label], confidence

            return None, confidence
        except Exception as e:
            print(f"얼굴 인식 오류: {e}")
            return None, 999
