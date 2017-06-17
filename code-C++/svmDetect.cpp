#include <opencv2\opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

//HOG描述符
HOGDescriptor detectHog;

//行人宽度缩放比例
double widthF = 0.5;

//行人高度缩放比例
double heightF = 0.9;


void svmDetect(Mat frameDetectGray, vector<Rect> &found)
{
	//HOG特征设置SVM分类器,使用默认的行人检测器
	detectHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//多尺度SVM检测
	//参数:(检测图像,输出结果,阈值,步长,扩充,尺寸变换,筛选)
	//理论上,不进行尺度设置也可以直接调整尺寸变换参数(1.07)进行检测,但是这样会大幅降低速度
	detectHog.detectMultiScale(frameDetectGray, found, 0, Size(4, 4), Size(0, 0), 1.2,2);

	for (int i = 0; i < found.size(); i++)
	{
		//检测到的行人一般而言轮廓都偏大，因此需要缩减
		found[i].x = cvRound(found[i].x + found[i].width * ((1 - widthF)/2));
		found[i].y = cvRound(found[i].y + found[i].height * ((1 - heightF) / 2));
		found[i].width = cvRound(found[i].width * widthF);
		found[i].height = cvRound(found[i].height * heightF);		
	}

}