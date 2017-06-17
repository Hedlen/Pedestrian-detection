#include <opencv2\opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

//定义参考帧数，和阈值.T1是运动检测阈值，T2是去除阴影的阈值
#define History 500
#define T1 16
#define T2 150

//MOG2背景建模函数，参数：历史参考帧，阈值，阴影检测
Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(History, T1, true);
//膨胀内核
Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(5, 10));
Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 10));


void moveDetect(Mat frameDetectGray, Mat &moveFound)
{
	moveFound = frameDetectGray.clone();

	//通过mog获得原始背景
	mog2->apply(moveFound, moveFound);

	//去除阴影，阴影的阈值大概在150附近，以此为阈值可以去掉阴影
	threshold(moveFound, moveFound, T2, 255, THRESH_BINARY);
	
	//中值滤波去除椒盐噪声
	//medianBlur(moveFound, moveFound, 5);	

	// 膨胀
	//dilate(moveFound, moveFound, kernel1);
	//morphologyEx(moveFound, moveFound, MORPH_CLOSE, kernel2);
}

