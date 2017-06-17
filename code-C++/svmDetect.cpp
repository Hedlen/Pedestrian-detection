#include <opencv2\opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

//HOG������
HOGDescriptor detectHog;

//���˿�����ű���
double widthF = 0.5;

//���˸߶����ű���
double heightF = 0.9;


void svmDetect(Mat frameDetectGray, vector<Rect> &found)
{
	//HOG��������SVM������,ʹ��Ĭ�ϵ����˼����
	detectHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//��߶�SVM���
	//����:(���ͼ��,������,��ֵ,����,����,�ߴ�任,ɸѡ)
	//������,�����г߶�����Ҳ����ֱ�ӵ����ߴ�任����(1.07)���м��,�����������������ٶ�
	detectHog.detectMultiScale(frameDetectGray, found, 0, Size(4, 4), Size(0, 0), 1.2,2);

	for (int i = 0; i < found.size(); i++)
	{
		//��⵽������һ�����������ƫ�������Ҫ����
		found[i].x = cvRound(found[i].x + found[i].width * ((1 - widthF)/2));
		found[i].y = cvRound(found[i].y + found[i].height * ((1 - heightF) / 2));
		found[i].width = cvRound(found[i].width * widthF);
		found[i].height = cvRound(found[i].height * heightF);		
	}

}