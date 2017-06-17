#include <opencv2\opencv.hpp>
#include <iostream>
#include <random>

using namespace std;
using namespace cv;

//������Ŀ�궨��
#define PARTICLE_NUMBER 100

//��������Ŀ
#define TRACK_NUM 50

//���ƶ��ж���ֵ
#define SIMILAR 0.5

//ģ�������ֵ
#define updateF 0.97

//���ٶ�����ֵ
#define deletionF 0.5


//��¼֡��
int FRAMECOUNT = 0;


//�����ݶ�ֱ��ͼ�õ��Ĳ���
int hist_size[] = {30, 32};
float hrange[] = {0, 180};
float srange[] = {0, 256};
const float *ranges[] = {hrange, srange};
int channels[] = {0, 1};




//������������
random_device rd; //���������
default_random_engine e(rd());  //���������  
normal_distribution<double> normalRandomX(0, 0.7);
normal_distribution<double> normalRandomY(0, 0.7);


// �������ӽṹ��
typedef struct particle
{
	int x;//��ǰ���ӵ�����(����)
	int y;
	Rect rect;
	double weight;		//��ǰ����Ȩֵ
}PARTICLE;


//����׷�ٶ���ṹ��
typedef struct trackObject
{
	int id; //׷�ٶ����id
	bool flag = false;
	int begain; //��ʼ֡
	int deleteN = 0; //ɾ������
	Rect origionRect;
	Rect currentRect; //����ǰ���ڵ�����
	Mat trackHist; //���������ֱ��ͼ������ƥ��
	PARTICLE particles[PARTICLE_NUMBER]; //׷�ٶ�������ӽṹ��
}TRACKOBJECT;


int ID = 0; //������������ID��
PARTICLE *pParticle; //ָ�����������ָ��

//׷�ٶ�������
TRACKOBJECT trackObjects[TRACK_NUM];

//Ȩֵ�������к���
int particle_decrease(const void *p1, const void *p2)
{
	PARTICLE* _p1 = (PARTICLE*)p1;
	PARTICLE* _p2 = (PARTICLE*)p2;
	if (_p1->weight<_p2->weight)
		return 1;
	else if (_p1->weight>_p2->weight)
		return -1;
	return 0;//��ȵ�����·���0
}


//ƥ����
void match(Mat frameHSV, vector<Rect> found)
{
	Rect foundRect;
	Mat foundImg;
	Mat foundHist;
	double similarF; //����ϵ��
	bool matchFlag = false;
	
	/*
	��׷�ٶ��������еĶ������ƥ��  
	ƥ�䵽�ˣ�˵���Ѿ�����׷��
	û��ƥ�䵽���ͽ��ü�������뵽׷�ٶ�����
	*/
	for (int i = 0; i < found.size(); i++)
	{
		foundRect = found[i];
		//����������ֱ��ͼ����
		foundImg = Mat(frameHSV, foundRect);
		calcHist(&foundImg, 1, channels, Mat(), foundHist, 2, hist_size, ranges);
		normalize(foundHist, foundHist,0,1,NORM_MINMAX,-1,Mat());
		
		//��ʼƥ��
		for (int j = 0; j < TRACK_NUM; j++)
		{
			if (trackObjects[j].flag == false) continue; // ����ö�����û�����ݾ������������ļ���
			else
			{
				similarF = 1 - compareHist(foundHist, trackObjects[j].trackHist, CV_COMP_BHATTACHARYYA);
				//������ƶȳ�����ֵ�Ϳ�����Ϊ��ƥ��ɹ���
				if (similarF > SIMILAR && (abs(foundRect.x - trackObjects[j].currentRect.x) < 0.5*foundRect.width))
				{
					//ƥ�䵽�˾ͽ��и���
					trackObjects[j].currentRect = foundRect;
					trackObjects[j].trackHist = foundHist;
					matchFlag = true;
					break;
				}
			}
		}

		//ѭ��������û���ҵ�ƥ���ϵģ����ڶ������½�һ��,����ɳ�ʼ��
		if (matchFlag == false)
		{
			// ���������ҵ�һ����λ�õ����� 
			int index; 
			for (index = 0; index < TRACK_NUM; index++)
			{
				if (trackObjects[index].flag == false) break;
			}
			
			//�ڸ�����λ�ô��½�һ��׷�ٶ���
			trackObjects[index].id = ID;
			trackObjects[index].flag = true;
			trackObjects[index].begain = FRAMECOUNT;
			trackObjects[index].origionRect = foundRect;
			trackObjects[index].currentRect = foundRect;
			trackObjects[index].trackHist = foundHist;
			ID++;

			//���ӳ�ʼ��
			pParticle = trackObjects[index].particles;//ָ���ʼ��ָ��particles����
			for (int k = 0; k < PARTICLE_NUMBER; k++)
			{
				//ѡ����׷��Ŀ����ο���Ϊ��ʼ���Ӿ��δ�
				pParticle[k].x = cvRound(foundRect.x + 0.5*foundRect.width); // ��ʼλ��Ϊ���˵�����
				pParticle[k].y = cvRound(foundRect.y + 0.5*foundRect.height);				
				pParticle[k].rect = foundRect;
				pParticle[k].weight = 0;//��ʼȨ�ض�Ϊ0
			}
		}
	}
}




//����׷�ٶ���
void updateParticles(Mat frameHSV)
{
	double sum;
	Mat particleImg;
	Mat particleHist;

	for (int i = 0; i < TRACK_NUM; i++)
	{
		if (trackObjects[i].flag == false) continue; //����δ��ʹ�ù�������λ��
		
		pParticle = trackObjects[i].particles; //����ָ��ָ�����鿪ʼ����

		//��׷�ٵĶ���PARTICLE_NUMBER�����ӽ��и���
		sum = 0.0;
		for (int j = 0; j < PARTICLE_NUMBER; j++)
		{
			//������������	
			if (pParticle[j].weight < 0.5) //�ջص�Ȩ�����ӣ�������һ֡��������ĸ���
			{
				pParticle[j].x = cvRound(trackObjects[i].currentRect.x + 0.5 * pParticle[j].rect.width + normalRandomX(e) * 0.5 * pParticle[j].rect.width);
				pParticle[j].y = cvRound(trackObjects[i].currentRect.y + 0.5 * pParticle[j].rect.height + normalRandomY(e) * 0.5 * pParticle[j].rect.height);
			}
			else //���ڸ�Ȩ������λ�ü�����������Χ
			{
				pParticle[j].x = cvRound(pParticle[j].x + normalRandomX(e) * 0.5 * pParticle[j].rect.width);
				pParticle[j].y = cvRound(pParticle[j].y + normalRandomY(e) * 0.5 * pParticle[j].rect.height);
			}
		

			//�����µ����Ӿ�������
			pParticle[j].rect.x = cvRound(pParticle[j].x - 0.5*pParticle[j].rect.width);
			pParticle[j].rect.y = cvRound(pParticle[j].y - 0.5*pParticle[j].rect.height);

			//���б߽���������ֹԽ�����
			if ((pParticle[j].rect.x + pParticle[j].rect.width) > frameHSV.cols) pParticle[j].rect.x = frameHSV.cols - pParticle[j].rect.width;
			if ((pParticle[j].rect.y + pParticle[j].rect.height) > frameHSV.rows) pParticle[j].rect.y = frameHSV.rows - pParticle[j].rect.height;
			if (pParticle[j].rect.x < 0) pParticle[j].rect.x = 0;
			if (pParticle[j].rect.y < 0) pParticle[j].rect.y = 0;
			
			//��������ӵ�ֱ��ͼ����
			particleImg = Mat(frameHSV, pParticle[j].rect);
			calcHist(&particleImg, 1, channels, Mat(), particleHist, 2, hist_size, ranges);
			normalize(particleHist, particleHist, 0, 1, NORM_MINMAX, -1, Mat());

			//�������ƶ�,��׷�ٶ����׷��ֱ��ͼ�Ƚϣ����ƶ���ΪȨ�ص�ֵ
			pParticle[j].weight = 1 - compareHist(particleHist, trackObjects[i].trackHist, CV_COMP_BHATTACHARYYA);
			
			//Ȩ���ۼ�
			sum += pParticle[j].weight;
		}

		//��Ȩ�ؽ�������
		qsort(pParticle, PARTICLE_NUMBER, sizeof(PARTICLE), &particle_decrease);
		
		////��ӡ���ƶ�
		//for (int j = 0; j<PARTICLE_NUMBER; j++)
		//{
		//	cout << pParticle[j].weight << "  ";
		//}
		//cout << endl << endl;


		if(pParticle->weight < deletionF) //���������ֵ�����ٸ��٣�������
		{
			trackObjects[i].flag = false;
			cout << "ID:" << trackObjects[i].id << endl;
			cout << "��ʼʱ��" << trackObjects[i].begain << endl;
			cout << "��ʼλ��" << trackObjects[i].origionRect.tl() << endl;
			cout << "����ʱ��" << FRAMECOUNT << endl;
			cout << "����λ��" << trackObjects[i].currentRect.tl() << endl;
			cout << endl << "****" << endl;

			continue;
		}
		else if(pParticle->weight > updateF)//���ڸ�����ֵ�������׷��Ŀ������
		{
			particleImg = Mat(frameHSV, pParticle->rect);
			calcHist(&particleImg, 1, channels, Mat(), particleHist, 2, hist_size, ranges);
			normalize(particleHist, particleHist, 0, 1, NORM_MINMAX, -1, Mat());
			trackObjects[i].trackHist = particleHist;
		}

		//�������ӵ�Ȩ�ع�һ��
		for (int j = 0; j<PARTICLE_NUMBER; j++)
		{
			pParticle[j].weight /= sum;
		}

		
		pParticle = trackObjects[i].particles; // ָ�����»ص���ʼ��ַ
		
		//���ٽ��
		Rect_<double> rectTrackingTemp(0.0, 0.0, 0.0, 0.0);
		int particleNum = (int)(PARTICLE_NUMBER * 0.01); //ȡһ���ָ�Ȩ�ص����Ӿ�ֵ��ΪԤ����
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
		
		//����Ŀ���������
		trackObjects[i].currentRect = Rect(rectTrackingTemp);

		//����Ȩ���ز�������
		PARTICLE newParticle[PARTICLE_NUMBER]; // ����һ���µ������������Դ���²���������
		int np, n=0;
		for (int j = 0; j < PARTICLE_NUMBER; j++)
		{
			np = cvRound(pParticle[j].weight * PARTICLE_NUMBER);
			for (int k = 0; k < np; k++) // Ȩ�ظߵ��²����ࣨѭ�������ࣩ
			{
				if (n == PARTICLE_NUMBER) break; //�������������������
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

		//��ʾ�������˶����
		pParticle = trackObjects[i].particles;
		
		for (int j = 0; j < PARTICLE_NUMBER; j++)
		{
			circle(frame, Point(pParticle[j].x, pParticle[j].y), 1, Scalar(255, 0, 0), 3); //���ӻ���ɫ��ʵ��Բ
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
	//���¼�¼��
	FRAMECOUNT++;

	//Ѱ���Ƿ��Ѿ��ڸ���������
	match(frameDetectHSV, svmFound);
	
	//��������
	updateParticles(frameDetectHSV);

	//���������˶����
	//drawParticleRect(frameDetect);

	//���ظ��ٵĽ��
	returnResult(trackResult);

}