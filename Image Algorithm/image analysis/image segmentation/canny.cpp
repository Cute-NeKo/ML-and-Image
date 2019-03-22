#include <opencv2/opencv.hpp>
#include<iostream>
#include<algorithm>

using namespace std;
using namespace cv;

Mat soble(Mat img);
Mat gaussianSmoothGray(Mat img);
Mat canny(Mat img);
Mat gray(Mat img);

int main()
{
   Mat img=imread("H://333.png");

   Mat img2=canny(img);


   //    imshow("image2",img2);
   //    imwrite("H://change.jpg",img3);
   waitKey(0);
   return 0;
}

Mat gray(Mat img){
   Mat changeImg(img.rows,img.cols,CV_8U);

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           uchar b=img.at<Vec3b>(i,j)[0];
           uchar g=img.at<Vec3b>(i,j)[1];
           uchar r=img.at<Vec3b>(i,j)[2];
           double gray= 0.3*r+0.59*g+0.11*b;
           changeImg.at<uchar>(i,j)=gray;

       }
   }
   return changeImg;
}

Mat soble(Mat img){

   Mat changeImg(img.rows,img.cols,CV_8UC3);

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){


           int filterX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
           int filterY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};int x=i;
           int y=j;
           if(x<=0){
               x=1;
           }
           if(x>=img.rows-1){
               x=img.rows-2;
           }
           if(y<=0){
               y=1;
           }
           if(y>=img.cols-1){
               y=img.cols-2;
           }

           double fx=0.0,fy=0.0;
           for(int m=-1;m<2;m++){
               for(int n=-1;n<2;n++){
                   fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[0];
                   fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[0];
                   fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[1];
                   fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[1];
                   fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[2];
                   fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[2];
               }
           }
           fx=fx/3.0;
           fy=fy/3.0;
           double f=fabs(fx)+fabs(fy);

           Vec3b vecf={(uchar)f,(uchar)f,(uchar)f};
           changeImg.at<Vec3b>(i,j)=vecf;

       }
   }
   return changeImg;

}

Mat gaussianSmoothGray(Mat img){
   Mat changeImg(img.rows,img.cols,CV_8U);

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int x=i;
           int y=j;
           if(x<=0){
               x=1;
           }
           if(x>=img.rows-1){
               x=img.rows-2;
           }
           if(y<=0){
               y=1;
           }
           if(y>=img.cols-1){
               y=img.cols-2;
           }

           double filter[3][3]={{1/16.0,2/16.0,1/16.0},{2/16.0,4/16.0,2/16.0},{1/16.0,2/16.0,1/16.0}};

           double b=0;
           for(int m=-1;m<2;m++){
               for(int n=-1;n<2;n++){
                   b+=img.at<uchar>(x+m,y+n)*filter[m+1][n+1];
               }
           }

           changeImg.at<uchar>(i,j)=b;

       }
   }
   return changeImg;
}

Mat canny(Mat img){
   Mat gx(img.rows,img.cols,CV_64F);
   Mat gy(img.rows,img.cols,CV_64F);
   Mat Mxy(img.rows,img.cols,CV_64F);
   Mat gx2;
   Mat gy2;
   Mat Axy(img.rows,img.cols,CV_64F);
   //灰度化
   Mat grayImg=gray(img);

   //高斯平滑
   Mat gaussImg=gaussianSmoothGray(grayImg);

   //求gx gy
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int x=i;
           int y=j;
           if(x<=0){
               x=1;
           }
           if(x>=img.rows-1){
               x=img.rows-2;
           }
           if(y<=0){
               y=1;
           }
           if(y>=img.cols-1){
               y=img.cols-2;
           }

           int filterX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
           int filterY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};


           double fx=0.0,fy=0.0;
           for(int m=-1;m<2;m++){
               for(int n=-1;n<2;n++){
                   fx+=filterX[m+1][n+1]*grayImg.at<uchar>(x+m,y+n);
                   fy+=filterY[m+1][n+1]*grayImg.at<uchar>(x+m,y+n);
               }
           }

           //            int filterX[2][2]={{-1,1},{-1,1}};
           //            int filterY[2][2]={{1,1},{-1,-1}};
           //            double fx=0,fy=0;
           //            for(int m=0;m<2;m++){
           //                for(int n=0;n<2;n++){
           //                    fx+=filterX[m][n]*grayImg.at<uchar>(x+m,y+n);
           //                    fy+=filterY[m][n]*grayImg.at<uchar>(x+m,y+n);
           //                }
           //            }

           gx.at<double>(i,j)=fx;
           gy.at<double>(i,j)=fy;
       }

   }

   //求梯度向量大小与方向

   cv::pow(gx,2,gx2);
   cv::pow(gy,2,gy2);
   cv::sqrt(gx2+gy2,Mxy);

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           Axy.at<double>(i,j)=atan2(gx.at<double>(i,j),gy.at<double>(i,j))*180/3.1415926;
       }
   }

   //非最大值抑制
   Mat Mxy2(img.rows,img.cols,CV_64F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int x=i;
           int y=j;
           if(x<=0){
               x=1;
           }
           if(x>=img.rows-1){
               x=img.rows-2;
           }
           if(y<=0){
               y=1;
           }
           if(y>=img.cols-1){
               y=img.cols-2;
           }

           double angle=Axy.at<double>(x,y);
           double dx,dy;
           if(angle<0){
               angle=angle+180;
           }
           if(angle<=22.5||angle>=157.5){
               dx=1;
               dy=0;
           }else if(angle>=22.5&&angle<=67.5){
               dx=1;
               dy=1;
           }else if(angle>=67.5&&angle<=112.5){
               dx=0;
               dy=1;
           }else{
               dx=-1;
               dy=1;
           }

           double M=Mxy.at<double>(x,y);
           double ML=Mxy.at<double>(x+dx,y+dy);
           double MR=Mxy.at<double>(x-dx,y-dy);
           if(M>ML&&M>MR){
               Mxy2.at<double>(i,j)=M;
           }

       }
   }

   double max=0;
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           if(Mxy2.at<double>(i,j)>max){
               max=Mxy2.at<double>(i,j);
           }
       }
   }
   cout<<max<<endl;

   //    for(int i=0;i<img.rows;i++){
   //        for(int j=0;j<img.cols;j++){
   //            cout<<Mxy2.at<double>(i,j)<<" ";
   //        }
   //    }

   //双阈值(普通法 不好)
   //    Mat gNH(img.rows,img.cols,CV_64F);
   //    Mat gNL(img.rows,img.cols,CV_64F);
   //    int nHist[1024];
   //    int nEdgeNum;
   //    int nMaxMag=0;
   //    for(int i=0;i<1024;i++){
   //        nHist[i]=0;
   //    }
   //    for(int i=0;i<img.rows;i++){
   //        for(int j=0;j<img.cols;j++){
   //            int ss=(int)Mxy2.at<double>(i,j);
   //            if(ss<1024){
   //                nHist[ss]++;
   //            }
   //        }
   //    }
   //    nEdgeNum=0;
   //    for(int i=1;i<1024;i++){
   //        if(nHist[i]!=0){
   //            nMaxMag=i;
   //        }
   //        nEdgeNum+=nHist[i];
   //    }

   //    int nThrHigh;
   //    int nThrLow;
   //    double dRateHigh=0.7;
   //    double dRateLow=0.5;
   //    int nHightCount=(int)(dRateHigh*nEdgeNum+0.5);
   //    int count=1;
   //    nEdgeNum=nHist[1];
   //    while((nEdgeNum<=nHightCount)&&(count<nMaxMag-1))
   //    {
   //        count++;
   //        nEdgeNum+=nHist[count];
   //    }
   //    nThrHigh=count;

   //    count=1;
   //    int nLowCount=(int)(nEdgeNum*dRateLow+0.5);
   //    nEdgeNum=nHist[1];
   //    while((nEdgeNum<=nLowCount)&&(count<nMaxMag-1))
   //    {
   //        count++;
   //        nEdgeNum+=nHist[count];
   //    }
   //    nThrLow=count;
   //    cout<<nThrHigh<<endl;
   //    cout<<nThrLow<<endl;

   //    for(int i=0;i<img.rows;i++){
   //        for(int j=0;j<img.cols;j++){
   //            if(Mxy2.at<double>(i,j)>nThrHigh){
   //                gNH.at<double>(i,j)=255;
   //            }
   //            if(Mxy2.at<double>(i,j)>nThrLow){
   //                gNL.at<double>(i,j)=255;
   //            }
   //        }
   //    }

   //otsu法求阈值
   //    Mat gNH(img.rows,img.cols,CV_64F);
   //    Mat gNL(img.rows,img.cols,CV_64F);
   //    int nHist[256];
   //    for(int i=0;i<256;i++){
   //        nHist[i]=0;
   //    }
   //    for(int i=0;i<img.rows;i++){
   //        for(int j=0;j<img.cols;j++){
   //            int ss=(int)Mxy2.at<double>(i,j);
   //            if(ss<256){
   //                nHist[ss]++;
   //            }
   //        }
   //    }
   //    double pHist[256]{0};
   //    for(int i=0;i<256;i++){
   //        pHist[i]=nHist[i]/(double(img.rows*img.cols));
   //    }
   //    double PHist[256]{0};
   //    for(int i=0;i<256;i++){
   //        for(int j=0;j<=i;j++){
   //            PHist[i]+=pHist[j];
   //        }
   //    }
   //    int MHist[256]{0};
   //    for(int i=0;i<256;i++){
   //        double ss=0;
   //        for(int j=0;j<=i;j++){
   //            ss+=(j*pHist[j]+0.5);
   //        }
   //        MHist[i]=int(ss);
   //    }
   //    int MG=MHist[255];
   //    long theta[256]{0};
   //    for(int i=0;i<256;i++){
   //        if((1-PHist[i])<=0){
   //            theta[i]=0;
   //        }else{
   //            double ss=pow((MG*PHist[i]-MHist[i]),2)/(PHist[i]*(1-PHist[i]));
   //            theta[i]=long(ss);
   //        }
   //    }

   //    int thre=0;
   //    max=0;
   //    for(int i=0;i<256;i++){
   //        if(theta[i]>max){
   //            max=theta[i];
   //            thre=i;
   //        }
   //    }
   //    cout<<thre<<endl;
   //    //    for(int i=0;i<256;i++){
   //    //        cout<<nHist[i]<<" ";
   //    //        if(i%20==0){
   //    //            cout<<endl;
   //    //        }
   //    //    }

   //自创法(形态学去噪)
   Mat gNH(img.rows,img.cols,CV_8U);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           if(Mxy2.at<double>(i,j)>255*0.25){
               gNH.at<uchar>(i,j)=255;
           }
       }
   }
   imshow("img",gNH);

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int x=i;
           int y=j;
           if(x<=1){
               x=2;
           }
           if(x>=img.rows-2){
               x=img.rows-3;
           }
           if(y<=1){
               y=2;
           }
           if(y>=img.cols-2){
               y=img.cols-3;
           }

           int k=0;
           for(int m=-2;m<3;m++){
               for(int n=-2;n<3;n++){
                   if(gNH.at<uchar>(x+m,y+n)==255){
                       k++;
                   }
               }
           }
           if(k<3){
               gNH.at<uchar>(x,y)=0;
           }
       }
   }



   imshow("img2",gNH);



   //        Mat Gxy(img.rows,img.cols,CV_8U);
   //        Mat Gxy2(img.rows,img.cols,CV_8U);
   //        for(int i=0;i<img.rows;i++){
   //            for(int j=0;j<img.cols;j++){
   //                Gxy.at<uchar>(i,j)=(uchar)Mxy.at<double>(i,j);
   //                Gxy2.at<uchar>(i,j)=(uchar)Mxy2.at<double>(i,j);
   //            }
   //        }
   //        imshow("img3",Gxy);
   //        imshow("img4",Gxy2);


}

