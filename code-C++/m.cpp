#include <opencv2\opencv.hpp>
#include <iostream>


using namespace cv;
using namespace std;

// 声明其他文件中的函数,包含所需要的各模块：运动检测，人体识别，滤波追踪
extern void setScale(Mat frame, Rect detectRect, double &scale, bool &selectFlag, bool &detectFlag);
extern void particleFilter(Mat frameDetectHSV, vector<Rect> &svmFound, Mat frameDetect, vector<Rect> &trackResult);
extern void moveDetect(Mat frameDetect, Mat &moveFound);
extern void svmDetect(Mat frameDetect, vector<Rect> &found);
extern void pedestrianJudge(Mat moveFound, vector<Rect> &svmFound);
void mouseSelect(int event, int x, int y, int flags, void* param);

//定义全局变量
//控制播放速度，降低计算量，每隔frequence帧计算一次（正常视频30帧/秒）
#define frequence 1

//检测视频文件名
#define videoName "18.mp4"



//播放器主界面Mat图像
Mat frame;
Mat frameOut;
Mat frameDetect;
Mat frameDetectGray;
Mat frameDetectHSV;

//鼠标开始选取区域标志
bool mouseFlag = false;
//检测区域选取完成标志
bool selectFlag = false;
//选取的矩形区域
Rect detectRect;
//扩招的矩形区域用与检测
Rect detectRectExtend;

//默认行人尺度为1.0，即原始大小
double scale = 1.0;

//开始检测标志位
bool detectFlag = false;

//运动检测结果二值图像
Mat moveFound; 

//存放SVM检测的结果
vector<Rect> svmFound;

//存放SVM检测的结果
vector<Rect> trackResult;

int main(int argc, unsigned char* argv[])
{
	VideoCapture cap(videoName); //加载待检测的视频

	//VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));
	if (!cap.isOpened())// 如果视频加载失败则返回
	{
		cout << "视频加载失败" << endl;
		return -1;
	}

	//读帧循环计数,用于控制播放速度
	int count = 0; 
	
	//创建主窗口，并给窗口绑定鼠标回调函数
	namedWindow("检测视频主播放窗口"); 
	setMouseCallback("检测视频主播放窗口", mouseSelect);

	while (true)
	{
		//循环计数,跳过frequence帧
		while(count < frequence)
		{
			//从视频流中读取一帧
			cap >> frame;

			//如果视频读取结束就退出
			if (frame.empty())
			{
				cout << "检测结束" << endl;
				return 1;
			}

			count++;		
		}
		//每次计数到frequence,清零
		count = 0;
 		
		//防止检测视频尺寸过小，将尺寸固定,这里调整不影响后续尺度设置
		resize(frame, frame, Size(1200, 800), 0, 0, INTER_LINEAR);
		
		//frame用于检测，frameOut用于画图，文字
		frameOut = frame.clone(); 

		//设置尺度
		if (selectFlag == true)
		{
			//选择完检测区域后,立即开始设置尺度大小
			//尺度设置完毕后，打开检测标志
			setScale(frame, detectRect, scale, selectFlag, detectFlag);
			
			//把检测区域向外扩充一个行人宽高(尺度变换后)距离
			int width = cvRound(64/scale);
			int height = cvRound(128/scale);
			detectRectExtend.width = detectRect.width + width;
			detectRectExtend.height = detectRect.height + height;
			detectRectExtend.x = detectRect.x - (int)(width/2.0);
			detectRectExtend.y = detectRect.y - (int)(height/2.0);
		}

		//检测流程
		if (detectFlag == true)
		{
			//在主窗口用红色画出矩形的检测区域
			rectangle(frameOut, detectRect, Scalar(0, 0, 255), 3);
			
			//减少计算量，只保留检测区域进行之后的检测
			frameDetect = frame(detectRectExtend);
			resize(frameDetect, frameDetect, Size(), scale, scale);
			cvtColor(frameDetect, frameDetectGray, COLOR_BGR2GRAY);
			cvtColor(frameDetect, frameDetectHSV, COLOR_BGR2HSV);

			//运动物体识别
			moveDetect(frameDetectGray, moveFound);
			//imshow("运动物体识别", moveFound);
			//SVM分类器识别人体
			svmDetect(frameDetectGray, svmFound);
			
			//将运动物体检测结果和行人识别结果综合判定
			pedestrianJudge(moveFound, svmFound);
			
			for (int i = 0; i < svmFound.size(); i++)
			{
				//rectangle(frameDetect, svmFound[i], Scalar(0,255,0), 2);
				//cout << svmFound[i].width << "   " << svmFound[i].height << endl;
			}
			
			//粒子滤波进行跟踪
			particleFilter(frameDetectHSV, svmFound, frameDetect, trackResult);

			for (int i = 0; i < trackResult.size(); i++)
			{
				rectangle(frameDetect, trackResult[i].tl(), trackResult[i].br(), Scalar(0, 255, 0), 3);
			}	

			//输出检测结果
			//resize(frameDetect, frameDetect, Size(detectRect.width, detectRect.height));
			imshow("检测结果", frameDetect);
		}

		//writer << frameDetect;
		imshow("检测视频主播放窗口", frameOut);
		if (waitKey(30*frequence) == 27)//等待30*frequence毫秒,中间如果按下ESC键,则退出循环,程序结束
		{
			break;
		}
	}
}



//鼠标回调函数，用于选取矩形区域
void mouseSelect(int event, int x, int y, int flags, void* param)
{
	//监听鼠标双击事件选点
	if (event == EVENT_LBUTTONDBLCLK)	
	{
		if (mouseFlag == false)	//第一次选点
		{
			detectRect.x = x;
			detectRect.y = y;
			mouseFlag = true;	//打开选点标志，表明已经选过第一个点			
		}
		else //第二次选点
		{
			//如果选中的第二点，不是右下角点，需要调整。
			//确保(detectRect.x,detectRect.y)为左上角点，（x,y）为右下角点
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

			detectRect.width = x - detectRect.x;	//通过两角点坐标计算矩形区域宽度和高度
			detectRect.height = y - detectRect.y;

			//关闭鼠标选取标志，打开选取完成标志
			mouseFlag = false;
			selectFlag = true;
		}
	}
}