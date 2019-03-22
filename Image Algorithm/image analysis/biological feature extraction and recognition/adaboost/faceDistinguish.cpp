#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;

#define ROW 20
#define COL 20

struct Scope{
   int row1;
   int col1;
   int row2;
   int col2;
};

struct Haar{
   int s;
   int t;
   vector<Scope> scopeList;
   vector<int> featureList;
};

int** getiimg(Mat img);
Haar* getHaarIndex();
int getFeatureValue(int** ,Scope ,double ,int,int);
Mat gray(Mat img);

int maisn(){
   Mat img=imread("h://hhh5.jpg");
   img=gray(img);
   //    rectangle(img,Point(100,100),Point(200,200),cvScalar(0, 0, 255), 3, 4, 0 );
   int** iimg=getiimg(img);
   Haar* haarIndex=getHaarIndex();

   vector<int> pList;
   vector<float> thetaList;
   vector<int> minIndexList;
   vector<float> aList;
   int p[8]={1,1,-1,1,1,1,1,1};
   float theta[8]={-4503,-3015,-1063,-3015,-3954,-1812,-3954,-919};
   int minIndex[8]={5939,2204,2210,2204,2204,2204,2204,2204};
   float a[8]={1.54263,1.56718,1.92398,3.3454,4.9296,8.88514,12.8904,32.7478};
   for(int i=0;i<8;i++){
       pList.push_back(p[i]);
       thetaList.push_back(theta[i]);
       minIndexList.push_back(minIndex[i]);
       aList.push_back(a[i]);
   }


   double aT=0;
   for(int i=0;i<aList.size();i++){
       aT+=aList[i];
   }
   aT/=2;
   cout<<aT<<endl;
   double width=20;
   double k=1;
   int kkk=0;
   while(k<img.rows||k<img.cols){
       for(int i=0;i<img.rows-width;i++){
           for(int j=0;j<img.cols-width;j++){
               double sum=0;
               for(int m=0;m<pList.size();m++){
                   int featureValue=getFeatureValue(iimg,haarIndex[0].scopeList[minIndexList[m]],k,i,j);
                   if(pList[m]==1&&featureValue<thetaList[m]){
                       sum+=aList[m];
                   }else if(pList[m]==-1&&featureValue>thetaList[m]){
                       sum+=aList[m];
                   }
               }
               //判断sum是否满足强分类器条件
//                if(sum>aT){
//                    rectangle(img,Point(i,j),Point(i+width,j+width),cvScalar(0, 0, 255), 1, 4, 0 );
//                    kkk++;
//                    break;
//                }
               cout<<sum<<" ";
           }
       }



       width*=1.2;
       k*=1.2;
   }
   cout<<kkk<<endl;
//    imshow("img",img);

   waitKey();
   return 0;
}

Haar* getHaarIndex(){
   Haar *haars=new Haar[5];
   haars[0].s=1;
   haars[0].t=2;
   haars[1].s=2;
   haars[1].t=1;
   haars[2].s=1;
   haars[2].t=3;
   haars[3].s=3;
   haars[3].t=1;
   haars[4].s=2;
   haars[4].t=2;
   for(int k=0;k<1;k++){
       for(int i=0;i<=ROW-haars[k].t;i++){
           for(int j=0;j<=COL-haars[k].s;j++){
               int q=(ROW-i)/haars[k].t;
               int p=(COL-j)/haars[k].s;
               for(int i2=1;i2<=q;i2++){
                   for(int j2=1;j2<=p;j2++){
                       Scope scope;
                       scope.row1=i;
                       scope.col1=j;
                       scope.row2=i+i2*haars[k].t-1;
                       scope.col2=j+j2*haars[k].s-1;
                       haars[k].scopeList.push_back(scope);
                   }
               }
           }
       }
   }
   return haars;

}

int getFeatureValue(int** iimg,Scope scope,double k,int rs,int cs){
   int feature;
   scope.row2=scope.row2*k;
   scope.col2=scope.col2*k;

   int r1=scope.row1-1,c1=scope.col1-1;  //判断是否小于0
   int r2=r1,c2=scope.col2;
   int r3=(scope.row1+scope.row2+1)/2-1,c3=c1;
   int r4=r3,c4=c2;
   int r5=scope.row2,c5=c1;
   int r6=r5,c6=c2;
   int f1=1,f2,f3,f4,f5,f6;
   if(r1<0){
       f1=0;
       f2=0;
   }else{
       f2=iimg[r2+rs][c2+cs];
   }
   if(c1<0){
       f1=0;f3=0;f5=0;
   }else{
       f3=iimg[r3+rs][c3+cs];f5=iimg[r5+rs][c5+cs];
   }
   if(f1){
       f1=iimg[r1+rs][c1+cs];
   }
   f4=iimg[r4+rs][c4+cs];f6=iimg[r6+rs][c6+cs];

   int fa=f4+f1-f2-f3;
   int fb=f6+f3-f4-f5;
   feature=fa-fb;
   return feature;
}

Mat gray(Mat img){
   Mat changeImg(img.rows,img.cols,CV_8UC3);

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           uchar b=img.at<Vec3b>(i,j)[0];
           uchar g=img.at<Vec3b>(i,j)[1];
           uchar r=img.at<Vec3b>(i,j)[2];
           double gray= 0.3*r+0.59*g+0.11*b;
           Vec3b bb={gray,gray,gray};
           changeImg.at<Vec3b>(i,j)=bb;

       }
   }
   return changeImg;
}




