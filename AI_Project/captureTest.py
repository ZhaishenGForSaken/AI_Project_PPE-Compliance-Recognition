import cv2

# 使用OpenCV捕捉视频
# 对于树莓派摄像头，通常使用0（或-1），也可以根据实际连接的摄像头调整
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
prev_frame_time = 0
new_frame_time = 0

while True:
    # 捕捉帧-by-帧
    ret, frame = cap.read()

    # 如果帧被正确读取，则ret为True
    if not ret:
        print("无法读取摄像头帧，退出...")
        break
    
    if (new_frame_time - prev_frame_time) > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0

    prev_frame_time = new_frame_time
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # 显示结果帧
    cv2.imshow('Frame', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
