from flask import Flask, render_template, Response,request
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
import webbrowser
from logging import root

run = Flask(__name__) 


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def video_detect():
    cap=cv2.VideoCapture(0)  
    value=0
    value_2=0
    detect=[]
    while True:  
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image  
        if not ret:  
            continue  
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  

        try:
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x,y,w,h) in faces_detected:  
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img = roi_gray.reshape((1,48,48,1))
                img = img /255.0

                max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                global predicted_emotion 
                predicted_emotion = emotions[max_index] 
                detect.append(predicted_emotion)

                if predicted_emotion:
                    value += 1
                    print(value)

                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if value>=50:
                    predicted_emotion="""You are in {} and i am suggest to you a songs""".format(predicted_emotion)
                    value_2 +=1
                    print(value_2)
                    cv2.putText(test_img, predicted_emotion, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        except:
            pass    

        resized_img = cv2.resize(test_img, (1000, 700))  
        ret, buffer = cv2.imencode('.jpg', resized_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # cv2.imshow('Facial emotion analysis ',resized_img)  

        if value_2==20:
        # if cv2.waitKey(10) == ord('s'):#wait until 's' key is pressed  
            cap.release()  
            cv2.destroyAllWindows()
            if detect[-1]=="neutral":       
                # url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0kSZqQE2zKYSisa5Vd33-WI"
                url = "https://open.spotify.com/track/7tYKa4wd7gL5LwcxidBPkG?si=f350bec799b04fe3"
                webbrowser.open(url)
            elif detect[-1]=="sad":
                url = "https://open.spotify.com/track/7tYKa4wd7gL5LwcxidBPkG?si=f350bec799b04fe3"

                webbrowser.open(url)
            elif detect[-1]=="happy":
                url = "https://open.spotify.com/track/7tYKa4wd7gL5LwcxidBPkG?si=f350bec799b04fe3"

                webbrowser.open(url)
            elif detect[-1]=="angry":
                url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0kIvJ_FDr8u8EEnPSoTXPcR"
                webbrowser.open(url)
            elif detect[-1]=="surprise":
                url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0nkF4xN7YbmNFtkQ6Qn6bf3"
                webbrowser.open(url)
            elif detect[-1]=="disgust":
                url = ": https://youtube.com/playlist?list=PL8Nkf7hoNm0m11La5RgqiugRQIHp7aVFl"
                webbrowser.open(url)
            elif detect[-1]=="fear":
                url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0mqhWoAnmYeg5bnSVQZ380q"
                webbrowser.open(url)
            break


def image_detect(files):
    value=0
    value_2=0
    detect=[]
    # file = askopenfile(filetypes =[('file selector', '*.jpg')])
    # print(str(file.name))
    c_img = cv2.imread(files)
    gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(c_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img = roi_gray.reshape((1,48,48,1))
        img = img /255.0

        max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                  
        predicted_emotion = emotions[max_index]  
        print(predicted_emotion)

        cv2.putText(c_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    resized_img = cv2.resize(c_img, (1000, 700))  
    cv2.imshow('Facial emotion analysis ',resized_img)
    cv2.imwrite("All_Emotions_Detection.jpg", resized_img)
    if cv2.waitKey(0) == ord('s'):
        cv2.destroyAllWindows()
    
    if predicted_emotion=="neutral":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0kSZqQE2zKYSisa5Vd33-WI"
        webbrowser.open(url)
    elif predicted_emotion=="sad":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0ldVRSUhH8UauQ2Z4BvZVis"
        webbrowser.open(url)
    elif predicted_emotion=="happy":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0m4vO2_P69Phxe4undoNYKZ"
        webbrowser.open(url)
    elif predicted_emotion=="angry":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0kIvJ_FDr8u8EEnPSoTXPcR"
        webbrowser.open(url)
    elif predicted_emotion=="surprise":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0nkF4xN7YbmNFtkQ6Qn6bf3"
        webbrowser.open(url)
    elif predicted_emotion=="disgust":
        url = ": https://youtube.com/playlist?list=PL8Nkf7hoNm0m11La5RgqiugRQIHp7aVFl"
        webbrowser.open(url)
    elif predicted_emotion=="fear":
        url = "https://youtube.com/playlist?list=PL8Nkf7hoNm0mqhWoAnmYeg5bnSVQZ380q"
        webbrowser.open(url)

@run.route('/')
def home():
    return render_template('index.html')

@run.route('/start')    
def cam():
    return render_template('start.html')

@run.route('/camera')
def cam_start():
    return render_template('cam.html')

@run.route('/video_feed')
def video_feed():
    return Response(video_detect(), mimetype='multipart/x-mixed-replace; boundary=frame')    

@run.route('/pic_start')    
def pic():  
    return render_template('vid_s.html')

@run.route('/pic', methods=['post'])
def pic_start():
    import os
    from werkzeug.utils import secure_filename
    file=request.files['file']
    files=file.filename
    print(files)   
    basepath = os.path.dirname(__file__)
    print(basepath)
    file_path = os.path.join(basepath, 'upload',secure_filename(file.filename))
    file.save(file_path)   
    image_detect(file_path)
    return render_template('video.html')

# @run.route('/pic_feed')
# def pic_feed():
#     return Response(image_detect(), mimetype='multipart/x-mixed-replace; boundary=frame')    


run.run(debug=True)    

