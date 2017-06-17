#include <opencv2\opencv.hpp>
#include <iostream>
#include <random>

using namespace std;
using namespace cv;

//粒子数目宏定义
#define PARTICLE_NUMBER 100

//最大跟踪数目
#define TRACK_NUM 50

//相似度判断阈值
#define SIMILAR 0.5

//模板更新阈值
#define updateF 0.97

//跟踪丢弃阈值
#define deletionF 0.5


//记录帧数
int FRAMECOUNT = 0;


//计算梯度直方图用到的参数
int hist_size[] = {30, 32};
float hrange[] = {0, 180};
float srange[] = {0, 256};
const float *ranges[] = {hrange, srange};
int channels[] = {0, 1};




//生成随机数相关
random_device rd; //随机数种子
default_random_engine e(rd());  //随机数引擎  
normal_distribution<double> normalRandomX(0, 0.7);
normal_distribution<double> normalRandomY(0, 0.7);


// 定义粒子结构体
typedef struct particle
{
	int x;//当前粒子的坐标(中心)
	int y;
	Rect rect;
	double weight;		//当前粒子权值
}PARTICLE;


//定义追踪对象结构体
typedef struct trackObject
{
	int id; //追踪对象的id
	bool flag = false;
	int begain; //开始帧
	int deleteN = 0; //删除计数
	Rect origionRect;
	Rect currentRect; //对象当前所在的区域
	Mat trackHist; //对象的特征直方图，用于匹配
	PARTICLE particles[PARTICLE_NUMBER]; //追踪对象的粒子结构体
}TRACKOBJECT;


int ID = 0; //给检测对象分配的ID号
PARTICLE *pParticle; //指向粒子数组的指针

//追踪对象数组
TRACKOBJECT trackObjects[TRACK_NUM];

//权值降序排列函数
int particle_decrease(const void *p1, const void *p2)
{
	PARTICLE* _p1 = (PARTICLE*)p1;
	PARTICLE* _p2 = (PARTICLE*)p2;
	if (_p1->weight<_p2->weight)
		return 1;
	else if (_p1->weight>_p2->weight)
		return -1;
	return 0;//相等的情况下返回0
}


//匹配结果
void match(Mat frameHSV, vector<Rect> found)
{
	Rect foundRect;
	Mat foundImg;
	Mat foundHist;
	double similarF; //相似系数
	bool matchFlag = false;
	
	/*
	与追踪对象数组中的对象进行匹配  
	匹配到了，说明已经正在追踪
	没有匹配到，就将该检测结果加入到追踪队列中
	*/
	for (int i = 0; i < found.size(); i++)
	{
		foundRect = found[i];
		//计算检测结果的直方图特征
		foundImg = Mat(frameHSV, foundRect);
		calcHist(&foundImg, 1, channels, Mat(), foundHist, 2, hist_size, ranges);
		normalize(foundHist, foundHist,0,1,NORM_MINMAX,-1,Mat());
		
		//开始匹配
		for (int j = 0; j < TRACK_NUM; j++)
		{
			if (trackObjects[j].flag == false) continue; // 如果该对象中没有数据就跳过接下来的计算
			else
			{
				similarF = 1 - compareHist(foundHist, trackObjects[j].trackHist, CV_COMP_BHATTACHARYYA);
				//如果相似度超过阈值就可以认为是匹配成功了
				if (similarF > SIMILAR && (abs(foundRect.x - trackObjects[j].currentRect.x) < 0.5*foundRect.width))
				{
					//匹配到了就进行更新
					trackObjects[j].currentRect = foundRect;
					trackObjects[j].trackHist = foundHist;
					matchFlag = true;
					break;
				}
			}
		}

		//循环结束后都没有找到匹配上的，就在对象中新建一个,并完成初始化
		if (matchFlag == false)
		{
			// 在数组中找到一个空位置的索引 
			int index; 
			for (index = 0; index < TRACK_NUM; index++)
			{
				if (trackObjects[index].flag == false) break;
			}
			
			//在该索引位置处新建一个追踪对象
			trackObjects[index].id = ID;
			trackObjects[index].flag = true;
			trackObjects[index].begain = FRAMECOUNT;
			trackObjects[index].origionRect = foundRect;
			trackObjects[index].currentRect = foundRect;
			trackObjects[index].trackHist = foundHist;
			ID++;

			//粒子初始化
			pParticle = trackObjects[index].particles;//指针初始化指向particles数组
			for (int k = 0; k < PARTICLE_NUMBER; k++)
			{
				//选定的追踪目标矩形框，作为初始粒子矩形窗
				pParticle[k].x = cvRound(foundRect.x + 0.5*foundRect.width); // 初始位置为行人的中心
				pParticle[k].y = cvRound(foundRect.y + 0.5*foundRect.height);				
				pParticle[k].rect = foundRect;
				pParticle[k].weight = 0;//初始权重都为0
			}
		}
	}
}




//更新追踪对象
void updateParticles(Mat frameHSV)
{
	double sum;
	Mat particleImg;
	Mat particleHist;

	for (int i = 0; i < TRACK_NUM; i++)
	{
		if (trackObjects[i].flag == false) continue; //跳过未被使用过的数组位置
		
		pParticle = trackObjects[i].particles; //否则，指针指向数组开始更新

		//对追踪的对象PARTICLE_NUMBER个粒子进行更新
		sum = 0.0;
		for (int j = 0; j < PARTICLE_NUMBER; j++)
		{
			//更新粒子坐标	
			if (pParticle[j].weight < 0.5) //收回低权重粒子，放在上一帧检测结果中心附近
			{
				pParticle[j].x = cvRound(trackObjects[i].currentRect.x + 0.5 * pParticle[j].rect.width + normalRandomX(e) * 0.5 * pParticle[j].rect.width);
				pParticle[j].y = cvRound(trackObjects[i].currentRect.y + 0.5 * pParticle[j].rect.height + normalRandomY(e) * 0.5 * pParticle[j].rect.height);
			}
			else //对于高权重粒子位置继续落在其周围
			{
				pParticle[j].x = cvRound(pParticle[j].x + normalRandomX(e) * 0.5 * pParticle[j].rect.width);
				pParticle[j].y = cvRound(pParticle[j].y + normalRandomY(e) * 0.5 * pParticle[j].rect.height);
			}
		

			//计算新的粒子矩形区域
			pParticle[j].rect.x = cvRound(pParticle[j].x - 0.5*pParticle[j].rect.width);
			pParticle[j].rect.y = cvRound(pParticle[j].y - 0.5*pParticle[j].rect.height);

			//进行边界修正，防止越界错误
			if ((pParticle[j].rect.x + pParticle[j].rect.width) > frameHSV.cols) pParticle[j].rect.x = frameHSV.cols - pParticle[j].rect.width;
			if ((pParticle[j].rect.y + pParticle[j].rect.height) > frameHSV.rows) pParticle[j].rect.y = frameHSV.rows - pParticle[j].rect.height;
			if (pParticle[j].rect.x < 0) pParticle[j].rect.x = 0;
			if (pParticle[j].rect.y < 0) pParticle[j].rect.y = 0;
			
			//计算该粒子的直方图特征
			particleImg = Mat(frameHSV, pParticle[j].rect);
			calcHist(&particleImg, 1, channels, Mat(), particleHist, 2, hist_size, ranges);
			normalize(particleHist, particleHist, 0, 1, NORM_MINMAX, -1, Mat());

			//计算相似度,与追踪对象的追踪直方图比较，相似度作为权重的值
			pParticle[j].weight = 1 - compareHist(particleHist, trackObjects[i].trackHist, CV_COMP_BHATTACHARYYA);
			
			//权重累加
			sum += pParticle[j].weight;
		}

		//对权重降序排列
		qsort(pParticle, PARTICLE_NUMBER, sizeof(PARTICLE), &particle_decrease);
		
		////打印相似度
		//for (int j = 0; j<PARTICLE_NUMBER; j++)
		//{
		//	cout << pParticle[j].weight << "  ";
		//}
		//cout << endl << endl;


		if(pParticle->weight < deletionF) //如果低于阈值，则不再跟踪，输出结果
		{
			trackObjects[i].flag = false;
			cout << "ID:" << trackObjects[i].id << endl;
			cout << "开始时间" << trackObjects[i].begain << endl;
			cout << "开始位置" << trackObjects[i].origionRect.tl() << endl;
			cout << "结束时间" << FRAMECOUNT << endl;
			cout << "结束位置" << trackObjects[i].currentRect.tl() << endl;
			cout << endl << "****" << endl;

			continue;
		}
		else if(pParticle->weight > updateF)//高于更新阈值，则更新追踪目标特征
		{
			particleImg = Mat(frameHSV, pParticle->rect);
			calcHist(&particleImg, 1, channels, Mat(), particleHist, 2, hist_size, ranges);
			normalize(particleHist, particleHist, 0, 1, NORM_MINMAX, -1, Mat());
			trackObjects[i].trackHist = particleHist;
		}

		//所有粒子的权重归一化
		for (int j = 0; j<PARTICLE_NUMBER; j++)
		{
			pParticle[j].weight /= sum;
		}

		
		pParticle = trackObjects[i].particles; // 指针重新回到起始地址
		
		//跟踪结果
		Rect_<double> rectTrackingTemp(0.0, 0.0, 0.0, 0.0);
		int particleNum = (int)(PARTICLE_NUMBER * 0.01); //取一部分高权重的粒子均值作为预测结果
		if (particleNum < 1)  particleNum = 1;
		for (int j = 0; j < particleNum; j++)
		{
			rectTrackingTemp.x += cvRound(pParticle[j].rect.x);
			rectTrackingTemp.y += cvRound(pParticle[j].rect.y);
			rectTrackingTemp.width = pParticle[j].rect.width;
			rectTrackingTemp.height = pParticle[j].rect.height;
		}
		
		rectTrackingTemp.x /= particleNum;
		rectTrackingTemp.y /= particleNum;
		
		//创建目标矩形区域
		trackObjects[i].currentRect = Rect(rectTrackingTemp);

		//根据权重重采样粒子
		PARTICLE newParticle[PARTICLE_NUMBER]; // 创建一个新的粒子数组用以存放新采样的粒子
		int np, n=0;
		for (int j = 0; j < PARTICLE_NUMBER; j++)
		{
			np = cvRound(pParticle[j].weight * PARTICLE_NUMBER);
			for (int k = 0; k < np; k++) // 权重高的新采样多（循环次数多）
			{
				if (n == PARTICLE_NUMBER) break; //新粒子数组采样完跳出
				newParticle[n++] = pParticle[j];
			}		
		}
		while (n < PARTICLE_NUMBER)
			newParticle[n++] = pParticle[0];
		
		for (int k = 0; k<PARTICLE_NUMBER; k++)
			pParticle[k] = newParticle[k];
	}
}



void drawParticleRect(Mat frame)
{
	for (int i = 0; i < TRACK_NUM; i++)
	{
		if (trackObjects[i].flag == false) continue;

		//显示各粒子运动结果
		pParticle = trackObjects[i].particles;
		
		for (int j = 0; j < PARTICLE_NUMBER; j++)
		{
			circle(frame, Point(pParticle[j].x, pParticle[j].y), 1, Scalar(255, 0, 0), 3); //粒子画蓝色的实心圆
		}	
		
		imshow("1", frame);
		waitKey(1);
	}
}

void returnResult(vector<Rect> &trackResult)
{
	trackResult.clear();

	for (int i = 0; i < TRACK_NUM; i++)
	{
		if (trackObjects[i].flag == false) continue;

		trackResult.push_back(trackObjects[i].currentRect);

	}
}


void particleFilter(Mat frameDetectHSV, vector<Rect> &svmFound, Mat frameDetect, vector<Rect> &trackResult)
{
	//更新记录数
	FRAMECOUNT++;

	//寻找是否已经在跟踪向量中
	match(frameDetectHSV, svmFound);
	
	//更新粒子
	updateParticles(frameDetectHSV);

	//绘制粒子运动结果
	//drawParticleRect(frameDetect);

	//返回跟踪的结果
	returnResult(trackResult);

}