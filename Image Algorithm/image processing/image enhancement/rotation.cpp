#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

Mat getRectFIlter(int r,int c,int angle);

int main(){
   //    Mat img=imread("h:\\R~Z$03)H%)~`2(G3$HW9}_W.png");
   //    imshow("img",img);
   //    int centerR=img.rows/2;
   //    int centerC=img.cols/2;
   //    Mat rotationMat=getRotationMatrix2D(Point2f(centerR,centerC),45,1.0);
   //    Mat img2;
   //    warpAffine(img,img2,rotationMat,img.size());
   //    imshow("img2",img2);

//    Mat img=getRectFIlter(3,20,90);
//    imshow("img",img);

   Mat img=imread("H:\\013.jpg");
   copyMakeBorder(img,img,200,200,200,200, BORDER_REFLECT_101);
   imshow("img",img);
   waitKey();
   return 0;
}

Mat getRectFIlter(int r,int c,int angle){
   int size=r>c?r:c;
   int borderAll=10;
   if(size%2==0){
       size+=(borderAll+1);
   }else{
       size+=borderAll;
   }
   Mat baseImg(size,size,CV_32F,Scalar(0));
   int topR=(size-r)/2,topC=(size-c)/2;
   int buttonR=topR+r,buttonC=topC+c;
   for(int i=topR;i<=buttonR;i++){
       for(int j=topC;j<buttonC;j++){
           baseImg.at<float>(i,j)=1;
       }
   }

   int centerR=size/2;
   int centerC=size/2;
   Mat rotationMat=getRotationMatrix2D(Point2f(centerR,centerC),angle,1.0);
   Mat img2;
   warpAffine(baseImg,img2,rotationMat,baseImg.size());
   return img2;


}
