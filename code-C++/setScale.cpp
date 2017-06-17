#include <opencv2\opencv.hpp> //OpenCv2
#include <iostream>

using namespace cv;
using namespace std;


Mat scaleOutFrame;
Mat scaleFrame;
Rect scaleRect;

int scaleValue0 = 1.1;
int scaleValue1 = 0;

void setScale(Mat frame, Rect selectRect, double &scale, bool &selectFlag, bool &detectFlag)
{
	if (selectRect.width <= 50 || selectRect.height <= 50)
	{
		cout << "检测区域过小，无法完成尺度设置" << endl;
		selectFlag = false;
		return;
	}

	namedWindow("尺度设置");
	createTrackbar("尺度系数A", "尺度设置", &scaleValue0, 5);//尺度十分位数
	createTrackbar("尺度系数B", "尺度设置", &scaleValue1, 9);//尺度个位数

	//从原始图像中分离出选择的检测区域
	scaleFrame = frame(selectRect);

	scaleRect.x = 100;
	scaleRect.y = 50;

	while (true)
	{
		//根据滑动条位置返回的值,计算尺度
		scale = getTrackbarPos("尺度系数A", "尺度设置") + (getTrackbarPos("尺度系数B", "尺度设置") / 10.0);

		//边界框大小,根据训练集中行人标准大小(64*128),配合尺度,计算出实际大小
		scaleRect.width = (int)(64 / scale);
		scaleRect.height = (int)(128 / scale);

		scaleOutFrame = scaleFrame.clone();
		rectangle(scaleOutFrame, scaleRect, Scalar(0, 255, 0), 2); // 绘制边界框

		imshow("尺度设置", scaleOutFrame);
		//绘制完成后按ESC结束尺度设置步骤
		if (waitKey(30) == 27)
		{
			//改变标志位
			selectFlag = false;
			detectFlag = true;
			//销毁尺度设置窗口
			destroyWindow("尺度设置");
			return ;
		}
	}
}