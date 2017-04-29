import cv2
import numpy as np


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)



try:
    cap = cv2.VideoCapture('14.mp4')
    mog = cv2.createBackgroundSubtractorMOG2()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cc = 0
    while True:
        _, frame = cap.read()
        half_frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(half_frame, cv2.COLOR_BGR2GRAY)

        mask = mog.apply(gray)
        bin = mask

        mask = cv2.medianBlur(mask, 11)
        kernel = np.ones((16, 8), np.uint8)
        mask = cv2.dilate(mask, kernel)
        ret, mask = cv2.threshold(mask, 12, 255, cv2.THRESH_BINARY)

        image, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_frame = frame[y:(y+h), x:(x+w)]
            resize_frame = cv2.resize(contour_frame, (64*2, 128*2), interpolation=cv2.INTER_CUBIC)


            found, _ = hog.detectMultiScale(resize_frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

            if len(found)>0:
                for i in found:
                    cv2.rectangle(half_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(half_frame, str(len(found)), (int(x+w/2), int(y)), font, 2, (255, 0, 0), 1)
                    cv2.imwrite(str(cc)+'.jpg', resize_frame)
                    cc += 1


        cv2.imshow('mask', mask)
        cv2.imshow('contour', half_frame)
        k = cv2.waitKey(35) & 0xff

        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('frame.jpg', frame)


finally:
    cap.release()
    cv2.destroyAllWindows()