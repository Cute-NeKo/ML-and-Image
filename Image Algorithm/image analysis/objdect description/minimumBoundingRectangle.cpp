#include<iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(){
   Mat result;
   //1.查找轮廓
   //1.1查找轮廓前的预处理（灰度图，阈值化）
   Mat srcImg = imread("H:\\0NU[7T{PMN1X1T)J`3G[{(2.png", CV_LOAD_IMAGE_COLOR);
   Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
   morphologyEx(srcImg, srcImg, MORPH_OPEN, element);
   Mat copyImg = srcImg.clone();
   cvtColor(srcImg, srcImg, CV_BGR2GRAY);
   threshold(srcImg, srcImg, 100, 255, CV_THRESH_BINARY);
   srcImg=255-srcImg;
   vector <vector<Point>> contours;
   //1.2查找轮廓
   findContours(srcImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//最外层轮廓
   //1.3绘制所有轮廓
   drawContours(copyImg, contours, -1, Scalar(0, 255, 0), 1, 8);
   //*2.由轮廓确定正外接矩形及最小外接矩形
   //2.1 定义Rect类型的vector容器boundRect存放正外接矩形，初始化大小为contours.size()即轮廓个数
   vector<Rect> boundRect(contours.size());
   //*2.2 定义Rect类型的vector容器roRect存放最小外接矩形，初始化大小为contours.size()即轮廓个数
   vector<RotatedRect> roRect(contours.size());
   //2.3 遍历每个轮廓
   for (int i = 0; i < 1; i++)
   {
       //2.4 由轮廓（点集）确定出正外接矩形并绘制
       boundRect[i] = boundingRect(Mat(contours[i]));
       //2.4.1获得正外接矩形的左上角坐标及宽高
       int width = boundRect[i].width;
       int height = boundRect[i].height;
       int x = boundRect[i].x;
       int y = boundRect[i].y;
       //2.4.2用画矩形方法绘制正外接矩形
       Mat result_c(height,width,CV_8U);
       rectangle(copyImg, Rect(x, y, width, height), Scalar(255, 0, 0), 2, 8);

       for(int r=0;r<height;r++){
           for(int c=0;c<width;c++){
               result_c.at<uchar>(r,c)=srcImg.at<uchar>(y+r,x+c);
           }
       }
       cv::resize(result_c, result_c, cv::Size(20, 20), (0, 0), (0, 0), cv::INTER_LINEAR);
       Mat fin(28,28,CV_8U,Scalar(0));
       for(int r=4;r<4+20;r++){
           for(int c=4;c<4+20;c++){
               fin.at<uchar>(r,c)=result_c.at<uchar>(r-4,c-4);
           }
       }
       result=fin;
   }
   threshold(result, result, 100, 255, CV_THRESH_BINARY);

   imshow("轮廓和正外接矩形和最小外接矩形", copyImg);
   imshow("result", result);
   waitKey(0);
   return 0;
}
