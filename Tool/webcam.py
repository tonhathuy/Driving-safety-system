from cv2 import cv2

webcam = cv2.VideoCapture(0)
cv2.namedWindow('webcam')
img_counter = 0
while True:
    rect, frame = webcam.read()
    cv2.imshow('cam', frame)

    if not rect:
        break

    key = cv2.waitKey(1)

    if key % 256 == 27:
        # nút esc
        break

    if key % 256 == 32:
        # nút spacebar
        img_name = 'image_{}.png'.format(img_counter)
        cv2.imwrite(img_name, frame)
        print('{} capture'.format(img_name))
        img_counter += 1

webcam.release()
cv2.destroyAllWindows()
