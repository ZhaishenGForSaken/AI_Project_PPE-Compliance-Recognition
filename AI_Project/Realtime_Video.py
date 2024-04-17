import cv2 as cv
from Realtime_Process import *
import time
import smtplib
from email.mime.text import MIMEText
import requests
import json


def realtime_video(path_of_model, category_list, color_label, con_para=0.5):
    # detect_model = tf.saved_model.load(path_of_model)
    # detect_model = load_model(path_of_model)
    cap = cv.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_model = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    prev_frame_time = 0
    new_frame_time = 0
    i = 0
    while True:
        ret, frame = cap.read()
        new_frame_time = time.time()
        if (new_frame_time - prev_frame_time) > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
        else:
            fps = 0
        prev_frame_time = new_frame_time
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
        i += 1
        if not ret:
            break

        # processed_frame = classify_and_label(img=frame, model=detect_model, categories=category_list, con_para=con_para)
        # processed_frame = detect_and_draw_boxes_keras(img=frame, model=detect_model, category=category_list, con_para=con_para, color_label=color_label, face_model=face_model)
        # processed_frame, my_str = detect_and_draw_boxes_tflite(img=frame, path_of_model=path_of_model,
        #                                                        category=category_list,
        #                                                        con_para=con_para, color_label=color_label,
        #                                                        face_model=face_model)
        processed_frame, my_str = detect_and_draw_boxes_eyes_tflite(img=frame, path_of_model=path_of_model,
                                                               category=category_list,
                                                               con_para=con_para, color_label=color_label,
                                                               eye_model=eye_cascade)
        cv.imshow('Video Stream', processed_frame)
        if i == 100:
            # send_email('670072993@qq.com', 'DrunkApee@gmail.com', 'Zyf19980424', 'Raspberry Pi Mask Detection Result', my_str)
            send_email_through_IFTTT(f'https://maker.ifttt.com/trigger/AI_Project/json/with/key/KmRgl5IK2P6YuO-HHVEHg', my_str)
            i = 0
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def send_email(sender_email, receiver_email, password, subject, body):
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())
    server.quit()
    
def send_email_through_IFTTT(url, body):
    data = {
    "EventName": "AI_Project",
    "OccurredAt": "2024/04/09",
    "JsonPayload": body  # ?????,????????????????
    }
    response = requests.post(url, json=data)



