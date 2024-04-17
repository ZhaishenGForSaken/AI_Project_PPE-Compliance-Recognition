import cv2
import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
from matplotlib import pyplot as plt



def detect_and_draw_boxes_keras(img, model, category, con_para, face_model, color_label):

    faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            crop = img[y:y + h, x:x + w]
            crop = cv2.resize(crop, (128, 128))
            crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
            mask_result = model.predict(crop)
            label = mask_result.argmax()  # 0 for 'MASK', 1 for 'NO MASK'
            cv2.putText(img, category[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_label[label], 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), color_label[label], 1)
    else:
        print("No faces detected")

    return img

def detect_and_draw_boxes_eyes_tflite(img, path_of_model, category, con_para, eye_model, color_label):
    if img is not None:
        my_str = ""
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        interpreter = tflite.Interpreter(model_path=path_of_model)
        interpreter.allocate_tensors()
        input_details=interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        eyes = eye_model.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)


        if len(eyes) > 0:
            i = 1
            faces = infer_face_from_eyes(eyes)
            for (x, y, w, h) in faces:
                crop = img[y:y + h, x:x + w]
                crop = cv2.resize(crop, (128, 128))
                crop = np.reshape(crop, [1, 128, 128, 3]).astype(np.float32) / 255.0

                interpreter.set_tensor(input_details[0]['index'], crop)
                interpreter.invoke()
                mask_result = interpreter.get_tensor(output_details[0]['index'])

                label = mask_result.argmax()
                cv2.putText(img, category[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_label[label], 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color_label[label], 1)

                my_str += "Face "
                my_str += "{i}"
                i += 1
                if label == 0:
                    my_str += ": Not Mask Correctly.\n"
                elif label == 1:
                    my_str += " Mask Correctly.\n"
                else:
                    my_str += ": No Mask.\n"
        else:
            print("No faces detected")
            my_str = "No Faces Detected.\n"

        return img, my_str

def detect_and_draw_boxes_tflite(img, path_of_model, category, con_para, face_model, color_label):
    my_str = ""
    
    # 加载TensorFlow Lite模型
    interpreter = tflite.Interpreter(model_path=path_of_model)
    interpreter.allocate_tensors()

    # 获取模型的输入和输出细节
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 对图像进行人脸检测
    faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    if len(faces) > 0:
        i = 1
        for (x, y, w, h) in faces:
            crop = img[y:y + h, x:x + w]
            crop = cv2.resize(crop, (128, 128))
            crop = np.reshape(crop, [1, 128, 128, 3]).astype(np.float32) / 255.0

            # 使用TensorFlow Lite模型进行推断
            interpreter.set_tensor(input_details[0]['index'], crop)
            interpreter.invoke()
            mask_result = interpreter.get_tensor(output_details[0]['index'])

            label = mask_result.argmax()  # 0 for 'MASK', 1 for 'NO MASK'
            cv2.putText(img, category[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_label[label], 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), color_label[label], 1)
            
            my_str += "Face "
            my_str += "{i}"
            i += 1
            if label == 0:
                my_str += ": Mask.\n"
            else:
                my_str += ": No Mask.\n"
    else:
        print("No faces detected")
        my_str = "No Faces Detected.\n"
    
    return img, my_str


def classify_and_label(img, model, categories, con_para):
    img_resized = tf.image.resize(img, [128, 128])
    img_resized = tf.cast(img_resized, tf.float32)
    img_batch = tf.expand_dims(img_resized, 0)
    input_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    prediction = model(input_tensor)
    prob = prediction.numpy()[0][0]

    print(prob)
    class_id = int(prob > con_para)
    class_name = categories[class_id]
    confidence = prob if class_id == 1 else 1 - prob

    label = f'{class_name}: {confidence:.2f}'
    cv.putText(img, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

def infer_face_from_eyes(eye_detections):
    sorted_eyes = sorted(eye_detections, key=lambda eye: eye[0])

    # Placeholder function to check if two eyes make a pair
    def is_a_pair(eye1, eye2):
        same_level = abs(eye1[1] - eye2[1]) < eye1[3]
        reasonable_distance = eye2[0] - (eye1[0] + eye1[2]) < 4 * eye1[3]
        return same_level and reasonable_distance

    eye_pairs = []
    i = 0
    while i < len(sorted_eyes) - 1:
        if is_a_pair(sorted_eyes[i], sorted_eyes[i + 1]):
            eye_pairs.append((sorted_eyes[i], sorted_eyes[i + 1]))
            i += 2
        else:
            i += 1

    face_bounding_boxes = []
    for eye1, eye2 in eye_pairs:
        eye_center_x = (eye1[0] + eye2[0] + eye2[2]) // 2
        eye_center_y = (eye1[1] + eye1[3] // 2 + eye2[1] + eye2[3] // 2) // 2

        eye_distance = abs(eye2[0] - eye1[0])
        face_width = 2 * eye_distance

        face_height = int(2.5 * eye_distance)
        face_x = eye_center_x - face_width // 2
        face_y = eye_center_y - face_height // 2

        face_bounding_boxes.append((face_x, face_y, face_width, face_height))

    return face_bounding_boxes
