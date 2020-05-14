# import pyautogui
from keypress import PressKey,ReleaseKey, W, A, S, D
import time

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

print('down')
PressKey(W)
time.sleep(3)
ReleaseKey(W)
print('up')
PressKey(S)
time.sleep(2)
ReleaseKey(S)
# import numpy as np
# from PIL import ImageGrab
# import cv2
# import time
# import pyautogui
# from directkeys import ReleaseKey, PressKey, W, A, S, D, mouse_rb_press,mouse_lb_press, mouse_lb_release

# def process_img(image):
#     original_image = image
#     # convert to gray
#     processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # edge detection
#     processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
#     return processed_img

# def main():
    
#     for i in list(range(4))[::-1]:
#         print(i+1)
#         time.sleep(1)

#     last_time = time.time()
#     while True:
#         print('down')
#         PressKey(W)
#         time.sleep(3)
#         print('up')
#         PressKey(W)
#         time.sleep(3)
#         PressKey(W)
#         screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
#         #print('Frame took {} seconds'.format(time.time()-last_time))
#         last_time = time.time()
#         new_screen = process_img(screen)
#         cv2.imshow('window', new_screen)
#         #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

# if __name__=="__main__":
#     main()