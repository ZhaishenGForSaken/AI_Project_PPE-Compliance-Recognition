import cv2 as cv
import numpy as np
import tensorflow as tf
import threading
import queue
from keras.models import load_model

def multi_thread_video_capture_and_process(path_to_classification, catefory_list, color_label ,con_para=0.7):
    face_model = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    frame_queue = queue.Queue()
    result_queue = queue.Queue()
    model = load_model(path_to_classification)
    category = catefory_list

    def detect_and_draw_boxes_keras():
        while True:
            img = frame_queue.get()
            if img is None:  # Stop signal
                result_queue.put(None)
                break
            faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    crop = img[y:y + h, x:x + w]
                    crop = cv.resize(crop, (128, 128))
                    crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
                    mask_result = model.predict(crop)
                    label = mask_result.argmax()  # 0 for 'MASK', 1 for 'NO MASK'
                    cv.putText(img, category[label], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color_label[label], 2)
                    cv.rectangle(img, (x, y), (x + w, y + h), color_label[label], 1)
            else:
                print("No faces detected")
            result_queue.put(img)

    def capture_video_mt():
        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
            if not result_queue.empty():
                display_frame = result_queue.get()
                cv.imshow('Video Stream', display_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    threading.Thread(target=detect_and_draw_boxes_keras, daemon=True).start()

    capture_video_mt()
    cv.destroyAllWindows()


# def multi_thread_video_capture_and_process(path_to_model, category_list, con_para=0.5):
#     detect_fn = tf.saved_model.load(path_to_model)
#     frame_queue = queue.Queue()
#     result_queue = queue.Queue()
#
#     def detect_and_draw_boxes_mt():
#         while True:
#             if not frame_queue.empty():
#                 frame = frame_queue.get()
#                 input_tensor = tf.convert_to_tensor([frame])
#                 detections = detect_fn(input_tensor)
#
#                 boxes = detections['detection_boxes'][0].numpy()
#                 classes = detections['detection_classes'][0].numpy().astype(np.int32)
#                 scores = detections['detection_scores'][0].numpy()
#
#                 for i in range(len(scores)):
#                     if scores[i] > con_para:
#                         box = boxes[i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
#                         box = box.astype(np.int32)
#
#                         class_name = category_list.get(classes[i], '未知')
#                         cv.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
#                         label = f'{class_name}: {scores[i]:.2f}'
#                         cv.putText(frame, label, (box[1], box[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#
#                 result_queue.put(frame)
#
#     def capture_video_mt():
#         cap = cv.VideoCapture(0)
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_queue.put(frame)
#             if not result_queue.empty():
#                 display_frame = result_queue.get()
#                 cv.imshow('Video Stream', display_frame)
#
#             if cv.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         cap.release()
#
#     threading.Thread(target=detect_and_draw_boxes_mt, daemon=True).start()
#
#     capture_video_mt()
#     cv.destroyAllWindows()
