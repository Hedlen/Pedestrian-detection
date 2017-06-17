#include <opencv2\opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

//��ɫ���ص���ռ�ı���
#define proportion 0.15

Rect svmFoundRect;
Mat moveFoundMat;
int rows;
int cols;
int SUM;
double p;

//ƽ���߶ȡ����
#define meanF 0.2
double sumWidth = 0;
double sumHeight = 0;
int N = 0;
double meanWidth;
double meanHeight;



void pedestrianJudge(Mat moveFound, vector<Rect> &svmFound)
{
	//ͨ��ƽ����߱ȶ�
	for (int i = 0; i < svmFound.size(); i++)
	{
		sumWidth += svmFound[i].width;
		N++;
		meanWidth = sumWidth / N;

		if (meanWidth*(1 - meanF) >= svmFound[i].width || svmFound[i].width >= meanWidth*(1 + meanF))
		{
			//cout << meanWidth << "  " << svmFound[i].width << "���ϸ����" << endl;
			svmFound.erase(svmFound.begin() + i);

			// �ظ�ԭ�ȵ�ƽ��ֵ
			if (N > 10)
			{
				sumWidth -= svmFound[i].width;
				N--;
				meanWidth = sumWidth / N;
			}
		}
	}
	



	for (int i = 0; i < svmFound.size(); i++)
	{
		svmFoundRect = svmFound[i];
		moveFoundMat = moveFound(svmFoundRect);

		rows = moveFoundMat.rows;
		cols = moveFoundMat.cols;
		SUM = 0;

		for (int j = 0; j < rows; j++)
		{
			//ȡ���е��׵�ַ
			uchar* data = moveFoundMat.ptr<uchar>(j);
			for (int k = 0; k < cols; k++)
			{
				//ͳ�ư�ɫ���ص���Ŀ
				if (data[k] > 0)
				{
					SUM ++;
				}
			}
		}

		p = SUM / (rows * cols);

		//���δ�ﵽ����,ɾ��������
		if (p < proportion)
		{
			svmFound.erase(svmFound.begin() + i);
		}
	}

}