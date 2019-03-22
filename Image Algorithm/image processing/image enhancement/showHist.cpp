#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

double** getHist(Mat img);
double** normalizeHist(double** hist);

int mai3n()
{
   Mat img=imread("H://hhh.png");

   double** hist=getHist(img);


   hist=normalizeHist(hist);
   //    for(int i=0;i<3;i++){
   //        for(int j=0;j<256;j++){
   //            cout<<hist[i][j]<<" ";
   //        }
   //        cout<<endl;
   //    }

   cv::Mat histPic(1000, 1000, CV_8UC3, cv::Scalar(0, 0, 0));


   for(int i=0;i<256;i++){
       line(histPic,Point(i,256),Point(i,256-256*hist[0][i]),Scalar(255,0,0));
       line(histPic,Point(i+260,256),Point(i+260,256-256*hist[1][i]),Scalar(0,255,0));
       line(histPic,Point(i+520,256),Point(i+520,256-256*hist[2][i]),Scalar(0,0,255));
   }


   imshow("image",histPic);
   waitKey(0);
   return 0;
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
           hist[2][img.at<Vec3b>(i,j)[2]]++;
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



