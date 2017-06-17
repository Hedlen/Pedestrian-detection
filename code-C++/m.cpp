#include <opencv2\opencv.hpp>
#include <iostream>


using namespace cv;
using namespace std;

// ���������ļ��еĺ���,��������Ҫ�ĸ�ģ�飺�˶���⣬����ʶ���˲�׷��
extern void setScale(Mat frame, Rect detectRect, double &scale, bool &selectFlag, bool &detectFlag);
extern void particleFilter(Mat frameDetectHSV, vector<Rect> &svmFound, Mat frameDetect, vector<Rect> &trackResult);
extern void moveDetect(Mat frameDetect, Mat &moveFound);
extern void svmDetect(Mat frameDetect, vector<Rect> &found);
extern void pedestrianJudge(Mat moveFound, vector<Rect> &svmFound);
void mouseSelect(int event, int x, int y, int flags, void* param);

//����ȫ�ֱ���
//���Ʋ����ٶȣ����ͼ�������ÿ��frequence֡����һ�Σ�������Ƶ30֡/�룩
#define frequence 1

//�����Ƶ�ļ���
#define videoName "18.mp4"



//������������Matͼ��
Mat frame;
Mat frameOut;
Mat frameDetect;
Mat frameDetectGray;
Mat frameDetectHSV;

//��꿪ʼѡȡ�����־
bool mouseFlag = false;
//�������ѡȡ��ɱ�־
bool selectFlag = false;
//ѡȡ�ľ�������
Rect detectRect;
//���еľ�������������
Rect detectRectExtend;

//Ĭ�����˳߶�Ϊ1.0����ԭʼ��С
double scale = 1.0;

//��ʼ����־λ
bool detectFlag = false;

//�˶��������ֵͼ��
Mat moveFound; 

//���SVM���Ľ��
vector<Rect> svmFound;

//���SVM���Ľ��
vector<Rect> trackResult;

int main(int argc, unsigned char* argv[])
{
	VideoCapture cap(videoName); //���ش�������Ƶ

	//VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));
	if (!cap.isOpened())// �����Ƶ����ʧ���򷵻�
	{
		cout << "��Ƶ����ʧ��" << endl;
		return -1;
	}

	//��֡ѭ������,���ڿ��Ʋ����ٶ�
	int count = 0; 
	
	//���������ڣ��������ڰ����ص�����
	namedWindow("�����Ƶ�����Ŵ���"); 
	setMouseCallback("�����Ƶ�����Ŵ���", mouseSelect);

	while (true)
	{
		//ѭ������,����frequence֡
		while(count < frequence)
		{
			//����Ƶ���ж�ȡһ֡
			cap >> frame;

			//�����Ƶ��ȡ�������˳�
			if (frame.empty())
			{
				cout << "������" << endl;
				return 1;
			}

			count++;		
		}
		//ÿ�μ�����frequence,����
		count = 0;
 		
		//��ֹ�����Ƶ�ߴ��С�����ߴ�̶�,���������Ӱ������߶�����
		resize(frame, frame, Size(1200, 800), 0, 0, INTER_LINEAR);
		
		//frame���ڼ�⣬frameOut���ڻ�ͼ������
		frameOut = frame.clone(); 

		//���ó߶�
		if (selectFlag == true)
		{
			//ѡ�����������,������ʼ���ó߶ȴ�С
			//�߶�������Ϻ󣬴򿪼���־
			setScale(frame, detectRect, scale, selectFlag, detectFlag);
			
			//�Ѽ��������������һ�����˿��(�߶ȱ任��)����
			int width = cvRound(64/scale);
			int height = cvRound(128/scale);
			detectRectExtend.width = detectRect.width + width;
			detectRectExtend.height = detectRect.height + height;
			detectRectExtend.x = detectRect.x - (int)(width/2.0);
			detectRectExtend.y = detectRect.y - (int)(height/2.0);
		}

		//�������
		if (detectFlag == true)
		{
			//���������ú�ɫ�������εļ������
			rectangle(frameOut, detectRect, Scalar(0, 0, 255), 3);
			
			//���ټ�������ֻ��������������֮��ļ��
			frameDetect = frame(detectRectExtend);
			resize(frameDetect, frameDetect, Size(), scale, scale);
			cvtColor(frameDetect, frameDetectGray, COLOR_BGR2GRAY);
			cvtColor(frameDetect, frameDetectHSV, COLOR_BGR2HSV);

			//�˶�����ʶ��
			moveDetect(frameDetectGray, moveFound);
			//imshow("�˶�����ʶ��", moveFound);
			//SVM������ʶ������
			svmDetect(frameDetectGray, svmFound);
			
			//���˶���������������ʶ�����ۺ��ж�
			pedestrianJudge(moveFound, svmFound);
			
			for (int i = 0; i < svmFound.size(); i++)
			{
				//rectangle(frameDetect, svmFound[i], Scalar(0,255,0), 2);
				//cout << svmFound[i].width << "   " << svmFound[i].height << endl;
			}
			
			//�����˲����и���
			particleFilter(frameDetectHSV, svmFound, frameDetect, trackResult);

			for (int i = 0; i < trackResult.size(); i++)
			{
				rectangle(frameDetect, trackResult[i].tl(), trackResult[i].br(), Scalar(0, 255, 0), 3);
			}	

			//��������
			//resize(frameDetect, frameDetect, Size(detectRect.width, detectRect.height));
			imshow("�����", frameDetect);
		}

		//writer << frameDetect;
		imshow("�����Ƶ�����Ŵ���", frameOut);
		if (waitKey(30*frequence) == 27)//�ȴ�30*frequence����,�м��������ESC��,���˳�ѭ��,�������
		{
			break;
		}
	}
}



//���ص�����������ѡȡ��������
void mouseSelect(int event, int x, int y, int flags, void* param)
{
	//�������˫���¼�ѡ��
	if (event == EVENT_LBUTTONDBLCLK)	
	{
		if (mouseFlag == false)	//��һ��ѡ��
		{
			detectRect.x = x;
			detectRect.y = y;
			mouseFlag = true;	//��ѡ���־�������Ѿ�ѡ����һ����			
		}
		else //�ڶ���ѡ��
		{
			//���ѡ�еĵڶ��㣬�������½ǵ㣬��Ҫ������
			//ȷ��(detectRect.x,detectRect.y)Ϊ���Ͻǵ㣬��x,y��Ϊ���½ǵ�
			if (x < detectRect.x)
			{
				int tempX = x;
				x = detectRect.x;
				detectRect.x = tempX;
			}
			if (y < detectRect.y)
			{
				int tempY = y;
				y = detectRect.y;
				detectRect.y = tempY;
			}

			detectRect.width = x - detectRect.x;	//ͨ�����ǵ����������������Ⱥ͸߶�
			detectRect.height = y - detectRect.y;

			//�ر����ѡȡ��־����ѡȡ��ɱ�־
			mouseFlag = false;
			selectFlag = true;
		}
	}
}