from Realtime_Video import *
# from Realtime_Video_Mt import *

if __name__ == "__main__":
    path = "mobile_studentv2.tflite"  # for your own path of model
    # print(tf.__version__)
    # category = {0: 'masked', 1: 'not masked'}
    category = {0: 'not masked correctedly', 1: 'masked properly', 2: 'not masked'}
    color_label = {0: (0, 255, 0), 1: (255, 0, 0), 2:(0, 0, 255)}# for the categories
    multi_thread = False
    if multi_thread is False:
        realtime_video(path_of_model=path, category_list=category, color_label=color_label)
    else:
        multi_thread_video_capture_and_process(path_to_classification=path, color_label=color_label, catefory_list=category)
