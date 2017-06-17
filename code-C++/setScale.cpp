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
		cout << "��������С���޷���ɳ߶�����" << endl;
		selectFlag = false;
		return;
	}

	namedWindow("�߶�����");
	createTrackbar("�߶�ϵ��A", "�߶�����", &scaleValue0, 5);//�߶�ʮ��λ��
	createTrackbar("�߶�ϵ��B", "�߶�����", &scaleValue1, 9);//�߶ȸ�λ��

	//��ԭʼͼ���з����ѡ��ļ������
	scaleFrame = frame(selectRect);

	scaleRect.x = 100;
	scaleRect.y = 50;

	while (true)
	{
		//���ݻ�����λ�÷��ص�ֵ,����߶�
		scale = getTrackbarPos("�߶�ϵ��A", "�߶�����") + (getTrackbarPos("�߶�ϵ��B", "�߶�����") / 10.0);

		//�߽���С,����ѵ���������˱�׼��С(64*128),��ϳ߶�,�����ʵ�ʴ�С
		scaleRect.width = (int)(64 / scale);
		scaleRect.height = (int)(128 / scale);

		scaleOutFrame = scaleFrame.clone();
		rectangle(scaleOutFrame, scaleRect, Scalar(0, 255, 0), 2); // ���Ʊ߽��

		imshow("�߶�����", scaleOutFrame);
		//������ɺ�ESC�����߶����ò���
		if (waitKey(30) == 27)
		{
			//�ı��־λ
			selectFlag = false;
			detectFlag = true;
			//���ٳ߶����ô���
			destroyWindow("�߶�����");
			return ;
		}
	}
}