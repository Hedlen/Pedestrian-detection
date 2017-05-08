import cv2
import numpy as np


class PreDetector(object):
    def __init__(self, video_name):
        self.video = video_name

        cap = cv2.VideoCapture(self.video)
        _, self.first_frame = cap.read()
        cap.release()
        self.roi = []
        self.scale = 1

    def prepare(self):
        self.get_roi()
        self.adjust_scale()
        cv2.destroyAllWindows()
        return self.roi, self.scale

    def get_roi(self):

        def mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK and len(self.roi) < 2:
                self.roi.append([x, y])
                cv2.circle(self.first_frame, (x, y), 10, (0, 255, 0), 1)

        cv2.namedWindow("GET-ROI")
        cv2.setMouseCallback("GET-ROI", mouse_click)

        while True:
            cv2.imshow('GET-ROI', self.first_frame)
            if cv2.waitKey(20) & 0xFF == 27:
                raise Exception("获取ROI中断")
            if len(self.roi) >= 2:
                break

        cv2.rectangle(self.first_frame, (self.roi[0][0], self.roi[0][1]), (self.roi[1][0], self.roi[1][1]), (0, 255, 0))
        cv2.imshow('GET-ROI', self.first_frame)
        cv2.destroyAllWindows()

    def adjust_scale(self):

        def set_scale(scale):
            integer = cv2.getTrackbarPos("scale_integer", "adjust_scale")
            decimal = cv2.getTrackbarPos("scale_decimal", "adjust_scale")
            self.scale = integer + float(decimal/10)

        cv2.namedWindow("adjust_scale")
        cv2.createTrackbar("scale_integer", "adjust_scale", 1, 5, set_scale)
        cv2.createTrackbar("scale_decimal", "adjust_scale", 0, 10, set_scale)
        roi1_x, roi1_y = self.roi[0]
        roi2_x, roi2_y = self.roi[1]

        while True:
            scale_frame = self.first_frame[roi1_y:roi2_y, roi1_x:roi2_x].copy()
            y, x = scale_frame.shape[:2]

            center = (int(x / 2), int(y / 2))
            tl = (center[0] - int(50 * self.scale), center[1] - int(100 * self.scale))
            br = (center[0] + int(50 * self.scale), center[1] + int(100 * self.scale))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(scale_frame, "scale:%s"%self.scale, (center[0]-20, center[1]), font, 0.5, (0, 0, 255))
            cv2.rectangle(scale_frame, (tl[0], tl[1]), (br[0], br[1]), (255, 0, 0))

            cv2.imshow("adjust_scale", scale_frame)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break
            elif k == 27:
                raise Exception("调整scale中断")



class Detector(object):
    def __init__(self, video, roi, scale):
        self.video = video
        self.roi = roi
        self.scale = scale

    def detect(self, speed=2):
        cap = cv2.VideoCapture(self.video)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        count = 0
        try:
            while True:
                flag, frame = cap.read()
                if flag is False or frame is None:
                    print("End")

                if count % (6*speed) == 0:
                    dec_frame = frame[self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]]
                    dec_frame = cv2.resize(dec_frame, None, fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_CUBIC)
                    found, _ = hog.detectMultiScale(dec_frame)

                    self.draw_detections(dec_frame, found)

                cv2.imshow("frame", dec_frame)
                k = cv2.waitKey(30) & 0xFF
                if k == 27:
                    raise Exception("检测中断")
                elif k == ord('q'):
                    break
                count += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def draw_detections(self, img, rects, thickness = 4):
        for x, y, w, h in rects:
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)





if __name__ == "__main__":

    video_name = "14.mp4"

    pre = PreDetector(video_name)
    roi, scale = pre.prepare()

    det = Detector(video_name, roi, scale)
    det.detect(speed=2)



