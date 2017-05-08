import cv2
import numpy as np

def found_p(img_name):
    img = cv2.imread(img_name)

    # img = cv2.resize(img, (64*2, 128*2))

    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.medianBlur(img, 5, 0)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    found, w = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)
    # found, w = hog.detectMultiScale(img)

    draw_detections(img, found)

    cv2.namedWindow('img')
    def xy(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(x, y)
    cv2.setMouseCallback('img', xy)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def video():
    cap = cv2.VideoCapture("15.mp4")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cc = 0
    _, frame2 = cap.read()
    while True:
        _, frame = cap.read()
        frame = frame[110:619, 61:910]
        if cc % 12 == 1:

            frame2 = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
            found, _ = hog.detectMultiScale(frame2)
            draw_detections(frame2, found)

        cv2.imshow('mask', frame2)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite(str(cc)+'.jpg', frame)

        cc+=1


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 4):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)








if __name__ == '__main__':
    # found_p('91.jpg')
    # found_p('3.jpg')
    video()
    # a = cv2.imread("frame.jpg")
    # b = a[436:521, 404:477]
    # cv2.imwrite('img2.jpg', b)