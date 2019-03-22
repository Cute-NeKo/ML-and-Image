#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

double** getHist(Mat img);
double** normalizeHist(double** hist);
Mat histBalanced(Mat img);

int main()
{
   Mat img=imread("H://hhh.png");

   //    Mat grayImg(img.rows,img.cols,CV_8U);

   //    for(int i=0;i<img.rows;i++){
   //        for(int j=0;j<img.cols;j++){
   //            grayImg.at<uchar>(i,j)=uchar(0.11*img.at<Vec3b>(i,j)[0]+0.59*img.at<Vec3b>(i,j)[1]+0.3*img.at<Vec3b>(i,j)[2]);
   //        }
   //    }

   Mat changeImg=histBalanced(img);

   imshow("image",changeImg);
   waitKey(0);
   return 0;
}

Mat histBalanced(Mat img){
   int MN=img.cols*img.rows;
   Mat changeImg(img.rows,img.cols,CV_8UC3);
   double **hist=getHist(img); //得到直方图
   double map[3][256]; //像素均衡映射数组

   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           map[i][j]=0;
       }
   }

   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           for(int k=0;k<=j;k++){
               map[i][j]+=hist[i][k];
           }
           map[i][j]=map[i][j]*255/MN;

           if(map[i][j]<0){
               map[i][j]=0;
           }else if(map[i][j]>255){
               map[i][j]=255;
           }
       }
   }

   for(int i=0;i<changeImg.rows;i++){
       for(int j=0;j<changeImg.cols;j++){
           changeImg.at<Vec3b>(i,j)[0]=(uchar)map[0][img.at<Vec3b>(i,j)[0]];
           changeImg.at<Vec3b>(i,j)[1]=(uchar)map[1][img.at<Vec3b>(i,j)[1]];
           changeImg.at<Vec3b>(i,j)[2]=(uchar)map[2][img.at<Vec3b>(i,j)[2]];
       }
   }
   return changeImg;
}

double** getHist(Mat img){
   double** hist=new double*[3];
   for(int i=0;i<3;i++){
       hist[i]=new double[256];
   }
   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           hist[i][j]=0;
       }
   }
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           hist[0][img.at<Vec3b>(i,j)[0]]++;
           hist[1][img.at<Vec3b>(i,j)[1]]++;
           hist[2][img.at<Vec3b>(i,j)[1]]++;
       }
   }

   return hist;

}


double** normalizeHist(double** hist){
   int max=0;
   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           if(hist[i][j]>max){
               max=hist[i][j];
           }
       }
   }
   double ** hist2=hist;
   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           hist2[i][j]/=max;
       }
   }
   return hist2;
}


