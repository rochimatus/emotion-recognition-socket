from flask import Flask, render_template, Response, jsonify
import io
from flask_socketio import SocketIO, emit
from PIL import Image
import base64
# from flask_cors import CORS
import numpy as np
from engineio.payload import Payload
from camera import Video


Payload.max_decode_packets = 500
app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')
whole_process = Video()
# CORS(app)

if __name__ == '__main__':
    socketio.run(app)
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def image(data_image):
    # print("masuk")
    sbuf = io.StringIO()
    sbuf.write(data_image)
    # print("data")
    # print(data_image)
    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)
    arrnp = np.array(pimg)
    
    # print("pimg")
    # print(arrnp)
    ## converting RGB to BGR, as opencv standards
    imgencode, predictions = whole_process.preprocess(arrnp)
    # print(predictions)
    # frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2GRAY)
    # print("frame")
    # print(frame)
    # Process the image frame
    # frame = imutils.resize(frame, width=700)
    # frame = cv2.flip(frame, 1)
    # imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData
    # send_back = jsonify([stringData, predictions])
    # emit the frame back
    # print(type(predictions))
    # predictions_list = predictions.tolist() if len(predictions)> 0 else [] 
    # print(type(predictions_list))
    result = [stringData, predictions]
    emit('response_back', result)

    # emit('predictions', predictions_list)