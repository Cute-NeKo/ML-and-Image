#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

double** getHist(Mat img);
double** normalizeHist(double** hist);
Mat histBalanced(Mat img);

int main()
{
   Mat img=imread("H://hhh2.png");

       Mat grayImg(img.rows,img.cols,CV_8U);

       for(int i=0;i<img.rows;i++){
           for(int j=0;j<img.cols;j++){
               grayImg.at<uchar>(i,j)=uchar(0.11*img.at<Vec3b>(i,j)[0]+0.59*img.at<Vec3b>(i,j)[1]+0.3*img.at<Vec3b>(i,j)[2]);
           }
       }


   imshow("image",grayImg);
   waitKey(0);
   return 0;
}






