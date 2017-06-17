#include <opencv2\opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

//白色像素点所占的比重
#define proportion 0.15

Rect svmFoundRect;
Mat moveFoundMat;
int rows;
int cols;
int SUM;
double p;

//平均高度、宽度
#define meanF 0.2
double sumWidth = 0;
double sumHeight = 0;
int N = 0;
double meanWidth;
double meanHeight;



void pedestrianJudge(Mat moveFound, vector<Rect> &svmFound)
{
	//通过平均宽高比对
	for (int i = 0; i < svmFound.size(); i++)
	{
		sumWidth += svmFound[i].width;
		N++;
		meanWidth = sumWidth / N;

		if (meanWidth*(1 - meanF) >= svmFound[i].width || svmFound[i].width >= meanWidth*(1 + meanF))
		{
			//cout << meanWidth << "  " << svmFound[i].width << "不合格，清除" << endl;
			svmFound.erase(svmFound.begin() + i);

			// 回复原先的平均值
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
			//取该行的首地址
			uchar* data = moveFoundMat.ptr<uchar>(j);
			for (int k = 0; k < cols; k++)
			{
				//统计白色像素点数目
				if (data[k] > 0)
				{
					SUM ++;
				}
			}
		}

		p = SUM / (rows * cols);

		//如果未达到比例,删除这个结果
		if (p < proportion)
		{
			svmFound.erase(svmFound.begin() + i);
		}
	}

}