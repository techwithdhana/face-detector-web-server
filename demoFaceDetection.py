import time
import cv2
from flask import Flask, render_template, Response
detector = cv2.CascadeClassifier('Haarcascade files/haarcascade_frontalface_default.xml')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def gen():
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0)
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(imgGray, 1.3, 5)

            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                break

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
            if cv2.waitKey(50) & 0xFF == ord('s'):
                break
        else:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='192.168.43.128', port='5000', debug=True)
