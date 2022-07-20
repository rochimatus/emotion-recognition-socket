import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
model = load_model('./static/resource/model')
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels=['angry', 'disgust', 'happy', 'fear', 'sad', 'surprise', 'neutral']

dim = (48, 48)
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class Video():
    def predict_emotion(self, img):
        predictions = model.predict(img)
        return predictions

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        faces = face_detect.detectMultiScale(normalized, 1.3, 5)
        print("faces: " + str(len(faces)))
        predictions = []
        for x, y, w, h in faces:
            x1, y1 = x+w, y+h
            pict = normalized[y:y1, x:x1]
            pict = cv2.resize(pict, dim, interpolation=cv2.INTER_AREA)

            grayscale = cv2.cvtColor(pict, cv2.COLOR_RGB2GRAY)
            img = image.img_to_array(grayscale)
            img = np.expand_dims(img, axis=0)
            pict = np.vstack([img])
            predictions = self.predict_emotion(pict)
            # print(pict)
            print("predict")
            print(predictions)
            most_prediction = np.argmax(predictions)
            # most_prediction = 0

            # print(np.amax(predictions))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(normalized, (x, y), (x1, y1), (255, 0, 255), 2)
            cv2.putText(normalized, labels[most_prediction] + " " +str(np.amax(predictions)), (x, y), font, 1, (255, 255, 255))
        ret, jpg = cv2.imencode('.jpg', normalized)
        return jpg, predictions