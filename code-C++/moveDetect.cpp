#include <opencv2\opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

//����ο�֡��������ֵ.T1���˶������ֵ��T2��ȥ����Ӱ����ֵ
#define History 500
#define T1 16
#define T2 150

//MOG2������ģ��������������ʷ�ο�֡����ֵ����Ӱ���
Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(History, T1, true);
//�����ں�
Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(5, 10));
Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 10));


void moveDetect(Mat frameDetectGray, Mat &moveFound)
{
	moveFound = frameDetectGray.clone();

	//ͨ��mog���ԭʼ����
	mog2->apply(moveFound, moveFound);

	//ȥ����Ӱ����Ӱ����ֵ�����150�������Դ�Ϊ��ֵ����ȥ����Ӱ
	threshold(moveFound, moveFound, T2, 255, THRESH_BINARY);
	
	//��ֵ�˲�ȥ����������
	//medianBlur(moveFound, moveFound, 5);	

	// ����
	//dilate(moveFound, moveFound, kernel1);
	//morphologyEx(moveFound, moveFound, MORPH_CLOSE, kernel2);
}

