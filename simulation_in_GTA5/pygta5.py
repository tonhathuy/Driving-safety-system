# import numpy as np
# from PIL import ImageGrab
# import cv2
# import time

# def draw_lines(img, lines):
#     try:
#         for line in lines:
#             coords = line[0]
#             cv2.line(img, (coords[0],coords[1]), (coords[2], coords[3]), [0,255,0], 3)
#     except:
#         pass

# def ROI(img, vertices):
#     mask = np.zeros_like(img)
#     cv2.fillPoly(mask, vertices, 255)
#     masked = cv2.bitwise_and(img, mask)
#     return masked

# def proccess_img(original_img):
#     processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#     processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
#     processed_img = cv2.GaussianBlur(processed_img, (3,3), 0 )
#     vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
#     processed_img = ROI(processed_img, [vertices])

#     #  edges
#     lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 20, 15)
#     draw_lines(processed_img,lines)
#     return processed_img

# # def screen_record(): 
# last_time = time.time()
# while(True):
#     # 800x600 windowed mode
#     printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
#     new_screen = proccess_img(printscreen)
#     print('loop took {} seconds'.format(time.time()-last_time))
#     last_time = time.time()
#     cv2.imshow('window',cv2.cvtColor(new_screen, cv2.COLOR_BGR2RGB))
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
# # import numpy as np
# # from PIL import ImageGrab
# # import cv2

# # while(True):
# #     printscreen_pil =  ImageGrab.grab(bbox=(0,40,800,640))
# #     printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
# #     .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3)) 
# #     cv2.imshow('window',printscreen_numpy)
# #     if cv2.waitKey(25) & 0xFF == ord('q'):
# #         cv2.destroyAllWindows()
# #         break
import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W, A, S, D
import pyautogui


def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (3,3), 0 )
    vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
    processed_img = roi(processed_img, [vertices])

    #                       edges
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 20, 15)
    draw_lines(processed_img,lines)
    return processed_img


def main():
    last_time = time.time()
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(0,40, 800, 640)))
        new_screen = process_img(screen)
        print('Loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', new_screen)
        #cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()