#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include<iostream>
using namespace std;
using namespace cv;

//得到直方图
void showHist(cv::Mat img){
   double *hist=new double[256];
   for(int i=0;i<256;i++){
       hist[i]=0;
   }
   double step=1.0/250;
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           float a=img.at<uchar>(i,j);
           int index=int(a);
           hist[index]++;
       }
   }
   int max=0;
   for(int j=0;j<256;j++){
       if(hist[j]>max){
           max=hist[j];
       }
   }

   for(int j=0;j<256;j++){
       hist[j]/=max;
   }

   Mat histPic(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));


   for(int i=0;i<256;i++){
       line(histPic,Point(i,256),Point(i,256-256*hist[i]),Scalar(255,0,0));
   }


   imshow("image",histPic);
}

cv::Mat mkKernel(int ks, double sig, double th, double lm, double ps,double gama)
{
   int hks = (ks-1)/2;
   double theta = -th*CV_PI/180;
   double psi = ps*CV_PI/180;
   double del = 2.0/(ks-1);
   double lmbd = lm;
   double sigma = sig/ks;
   double x_theta;
   double y_theta;
   cv::Mat kernel(ks,ks, CV_32F);
   for (int y=-hks; y<=hks; y++)
   {
       for (int x=-hks; x<=hks; x++)
       {
           x_theta = x*del*cos(theta)+y*del*sin(theta);
           y_theta = -x*del*sin(theta)+y*del*cos(theta);
           kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(gama,2)*pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
       }
   }
   return kernel;
}

int kernel_size=21;
int pos_sigma= 10;
int pos_lm = 80;
int pos_th = 0;
int pos_psi = 90;
int pos_gama=4;
cv::Mat src_f;
cv::Mat dest;

void Process(int , void *)
{
   double sig = pos_sigma;
   double lm = 0.5+pos_lm/100.0;
   double th = pos_th;
   double ps = pos_psi;
   double gama=pos_gama;
   cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps,gama);
   cv::filter2D(src_f, dest, CV_32F, kernel);

   cv::Mat hhh;

   cv::pow(dest, 2.0, hhh);

   cv::normalize(hhh,hhh,0,255,CV_MINMAX);

   hhh.convertTo(hhh,CV_8U);
   equalizeHist(hhh,hhh);
   hhh.convertTo(hhh,CV_32F);
   cv::normalize(hhh,hhh,0,1,CV_MINMAX);

   showHist(hhh);

   cv::imshow("FFF",hhh);
   cv::imshow("Process window", hhh);
//     cv::imshow("Process window", dest);

   cv::Mat Lkernel(kernel_size*20, kernel_size*20, CV_32F);
   cv::resize(kernel, Lkernel, Lkernel.size());
   Lkernel /= 2.;
   Lkernel += 0.5;
   cv::imshow("Kernel", Lkernel);
   cv::Mat mag;
   cv::pow(dest, 2.0, mag);
   cv::imshow("Mag", mag);
//    cv::imwrite("C:\\Users\\NeKo\\Desktop\\gaborFilter.jpg",mag);
}


int main(int argc, char** argv)
{
   cv::Mat image = cv::imread("H:\\city7.png",1);
   cv::imshow("Src", image);
   cv::Mat src;
   cv::cvtColor(image, src, CV_BGR2GRAY);
   src.convertTo(src_f, CV_32F, 1.0/255, 0);
   if (!kernel_size%2)
   {
       kernel_size+=1;
   }
   cv::namedWindow("Process window", 1);
   cv::createTrackbar("Sigma", "Process window", &pos_sigma, kernel_size, Process);
   cv::createTrackbar("Lambda", "Process window", &pos_lm, 100, Process);
   cv::createTrackbar("Theta", "Process window", &pos_th, 180, Process);
   cv::createTrackbar("Psi", "Process window", &pos_psi, 360, Process);
    cv::createTrackbar("gama", "Process window", &pos_gama, 30, Process);
   Process(0,0);
   cv::waitKey(0);
   return 0;
}
