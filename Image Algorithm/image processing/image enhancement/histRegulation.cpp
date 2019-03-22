#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

Mat getHistRegulation(Mat img,Mat img2);
double** getHist(Mat img);

int main()
{
   Mat img=imread("H://hhh.png");
   Mat img2=imread("H://hhh2.png");
   imshow("image",img);
   Mat changeimg=getHistRegulation(img,img2);
   imshow("img2",changeimg);

   waitKey(0);
   return 0;
}

Mat getHistRegulation(Mat img,Mat img2){
   int MN=img.cols*img.rows;
   Mat changeImg(img.rows,img.cols,CV_8UC3);
   Mat tempMat(img.rows,img.cols,CV_64FC3);
   double **hist1=getHist(img);
   double **hist2=getHist(img2);
   double map1[3][256];
   double map2[3][256];
   double map3[3][256];

   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           map1[i][j]=0;
           map2[i][j]=0;
           map3[i][j]=-1;
       }
   }

   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           for(int k=0;k<=j;k++){
               map1[i][j]+=hist1[i][k];
               map2[i][j]+=hist2[i][k];
           }
           map1[i][j]=map1[i][j]*255/MN;
           map2[i][j]=map2[i][j]*255/MN;

           if(map1[i][j]<0){
               map1[i][j]=0;
           }else if(map1[i][j]>255){
               map1[i][j]=255;
           }

           if(map2[i][j]<0){
               map2[i][j]=0;
           }else if(map2[i][j]>255){
               map2[i][j]=255;
           }
       }
   }

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           tempMat.at<Vec3d>(i,j)[0]=map1[0][img.at<Vec3b>(i,j)[0]];
           tempMat.at<Vec3d>(i,j)[1]=map1[1][img.at<Vec3b>(i,j)[1]];
           tempMat.at<Vec3d>(i,j)[2]=map1[2][img.at<Vec3b>(i,j)[2]];
       }
   }


   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           map3[0][(int)map2[0][j]]=j;
           map3[1][(int)map2[1][j]]=j;
           map3[2][(int)map2[2][j]]=j;
       }
   }

   for(int i=0;i<3;i++){
       for(int j=0;j<256;j++){
           if(map3[i][j]==-1){
               int k=0;
               bool t=false;
               while(map3[i][j+k]==-1){
                   if(j+k>256){
                       k=-1;
                       t=true;
                   }
                   if(t){
                       k--;
                   }else{
                       k++;
                   }
               }
               map3[i][j]=map3[i][(j+k)%256];
           }
       }
   }


   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           changeImg.at<Vec3b>(i,j)[0]=(uchar)map3[0][(int)tempMat.at<Vec3d>(i,j)[0]];
           changeImg.at<Vec3b>(i,j)[1]=(uchar)map3[1][(int)tempMat.at<Vec3d>(i,j)[1]];
           changeImg.at<Vec3b>(i,j)[2]=(uchar)map3[2][(int)tempMat.at<Vec3d>(i,j)[2]];
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










