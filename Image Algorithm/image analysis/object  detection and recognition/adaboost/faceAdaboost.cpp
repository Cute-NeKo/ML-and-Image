#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include<QDir>

using namespace std;
using namespace cv;

#define NUM_THREADS 5

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

vector<int>* featureArray;
vector<int> labelList;
int sampleNum=0;
int featureNum=0;
float W[7089];


int** getiimg(Mat img);
void getHaar(int** iimg);
void adaBoost();
void quickSort(int low,int high,int k);
int median3(int left,int right,int k);
void swapReference(int m,int n);
void insertionSort( int left, int right,int k);
//void adaboostSon(float errorSample[sampleNum],
//                 float errorFeature[featureNum],
//                 float thetaSample[sampleNum],
//                 float thetaFeature[featureNum],
//                 int p[featureNum],vector<int> idx);

int main2()
{
   featureArray=new vector<int>[42000];


   string path="E:\\FaceDATA\\faces";
   QDir dir(QString::fromStdString(path));
   foreach(QFileInfo mfi ,dir.entryInfoList())
   {
       if(mfi.isFile())
       {
           string imgPath=path+"\\"+mfi.fileName().toStdString();
           Mat img=imread(imgPath);
           labelList.push_back(1);
           int **iimg=getiimg(img);
           getHaar(iimg);

           sampleNum++;
       }
   }
   path="E:\\FaceDATA\\nonfaces";
   QDir dir2(QString::fromStdString(path));
   foreach(QFileInfo mfi ,dir2.entryInfoList())
   {
       if(mfi.isFile())
       {
           string imgPath=path+"\\"+mfi.fileName().toStdString();
           Mat img=imread(imgPath);
           labelList.push_back(-1);
           int **iimg=getiimg(img);
           getHaar(iimg);
           sampleNum++;
       }
   }
   cout<<"数据读取完毕"<<endl;
   for(int i=0;i<42000;i++){
       cout<<featureArray[i][400]<<" ";
   }

   /***********************************************/
   adaBoost();

   waitKey(0);
   return 0;
}



int** getiimg(Mat img){
   int** iimg=new int*[img.rows];
   int** simg=new int*[img.rows];
   for(int i=0;i<img.rows;i++){
       iimg[i]=new int[img.cols];
       simg[i]=new int[img.cols];
   }
   for(int j=0;j<img.cols;j++){
       simg[0][j]=img.at<Vec3b>(0,j)[0];

       if(j==0){
           iimg[0][j]=img.at<Vec3b>(0,j)[0];
       }else{
           iimg[0][j]=iimg[0][j-1]+img.at<Vec3b>(0,j)[0];
       }
   }
   for(int i=1;i<img.rows;i++){
       simg[i][0]=simg[i-1][0]+img.at<Vec3b>(i,0)[0];
       iimg[i][0]=iimg[i-1][0]+img.at<Vec3b>(i,0)[0];
   }

   for(int i=1;i<img.rows;i++){
       for(int j=1;j<img.cols;j++){
           simg[i][j]=simg[i-1][j]+img.at<Vec3b>(i,j)[0];
           iimg[i][j]=iimg[i][j-1]+simg[i][j];
       }
   }

   return iimg;

}

void getHaar(int** iimg){
   featureNum=0;
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
   for(int k=0;k<2;k++){
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
                       int featue=0;
                       if(k==0){
                           int r1=i-1,c1=j-1;  //判断是否小于0
                           int r2=r1,c2=scope.col2;
                           int r3=r1+i2*haars[k].t/2,c3=c1;
                           int r4=r3,c4=c2;
                           int r5=scope.row2,c5=c1;
                           int r6=r5,c6=c2;
                           int f1=1,f2,f3,f4,f5,f6;
                           if(r1<0){
                               f1=0;
                               f2=0;
                           }else{
                               f2=iimg[r2][c2];
                           }
                           if(c1<0){
                               f1=0;f3=0;f5=0;
                           }else{
                               f3=iimg[r3][c3];f5=iimg[r5][c5];
                           }
                           if(f1){
                               f1=iimg[r1][c1];
                           }
                           f4=iimg[r4][c4];f6=iimg[r6][c6];

                           int fa=f4+f1-f2-f3;
                           int fb=f6+f3-f4-f5;
                           featue=fa-fb;

                       }else if(k==1){
                           int r1=i-1,c1=j-1;  //判断是否小于0
                           int r2=r1,c2=j2*haars[k].s/2;
                           int r3=r1,c3=scope.col2;
                           int r4=scope.row2,c4=c1;
                           int r5=r4,c5=c2;
                           int r6=r4,c6=c3;
                           int f1=1,f2,f3,f4,f5,f6;
                           if(r1<0){
                               f1=0;f2=0;f3=0;
                           }else{
                               f2=iimg[r2][c2];f3=iimg[r3][c3];
                           }
                           if(c1<0){
                               f1=0;f4=0;
                           }else{
                               f4=iimg[r4][c4];
                           }
                           if(f1){
                               f1=iimg[r1][c1];
                           }
                           f5=iimg[r5][c5];f6=iimg[r6][c6];
                           int fa=f5+f1-f2-f4;
                           int fb=f6+f2-f3-f5;
                           featue=fa-fb;

                       }else if(k==2){

                       }else if(k==3){

                       }else if(k==4){

                       }
                       //k=1 k=2 k=3 k=4之后在写


                       //                        haars[k].featureList.push_back(featue);
                       featureArray[featureNum].push_back(featue);
                       featureNum++;
                   }
               }
           }
       }
   }
   delete(haars);

}

void adaBoost(){
   //求权重并归一化
   int m=0,l=0;
   for(int i=0;i<sampleNum;i++){
       if(labelList[i]==1){
           l++;
       }else{
           m++;
       }
   }
   float fff=0;
   for(int i=0;i<sampleNum;i++){
       if(labelList[i]==1){
           W[i]=1/(2.0*l);
       }else{
           W[i]=1/(2.0*m);
       }
       fff+=W[i];
   }
   for(int i=0;i<sampleNum;i++){
       W[i]/=fff;
   }
   /***************************************/
   //计算每一个特征的分类器
   float TPlus=0,TMinus=0;
   for(int i=0;i<sampleNum;i++){
       if(labelList[i]==1){
           TPlus+=W[i];
       }else{
           TMinus+=W[i];
       }

   }
   /****************************************/
   vector<int> pList;
   vector<float> thetaList;
   vector<int> minIndexList; //最优分类器索引
   vector<float> aList;

   while(true){
       float errorSample[sampleNum];
       float errorFeature[featureNum];
       float thetaSample[sampleNum];
       float thetaFeature[featureNum];
       int p[featureNum]; //设置分类器的方向（小于分类器阈值的是正例还是反例）
       vector<int> idx(sampleNum);
       for(int k=0;k<featureNum;k++){

           //排序（记录索引）
           iota(idx.begin(),idx.end(),0);
           sort(idx.begin(),idx.end(), [&](int i1, int i2){return featureArray[k][i1]<featureArray[k][i2];});
           //寻找本特征的最小误差分类器
           float minError=10;int minIndex;
           for(int i=0;i<sampleNum;i++){
               float SPlus=0,SMinus=0;

               for(int j=0;j<=i;j++){
                   if(labelList[idx[j]]==1){
                       SPlus+=W[idx[j]];
                   }else{
                       SMinus+=W[idx[j]];
                   }
               }
               float e1=SPlus+(TMinus-SMinus);
               float e2=SMinus+(TPlus-SPlus);

               errorSample[i]=e1>e2?e2:e1;
               thetaSample[i]=featureArray[k][i];
               //找最小误差的划分点
               if(errorSample[i]<minError){
                   minError=errorSample[i];
                   minIndex=i;
               }
           }
           //            cout<<minError<<" ";

           thetaFeature[k]=thetaSample[minIndex];
           errorFeature[k]=minError;

           //确定分类器的方向
           int pp=0;
           for(int i=0;i<=minIndex;i++){
               if(labelList[idx[i]]==1){
                   pp++;
               }
           }
           p[k]=pp/float(minIndex)>0.5?1:-1;
           //            cout<<thetaFeature[k]<<" ";

           //计算分类器误差(经验证错误率等于minError)
           //        float error=0;
           //        for(int i=0;i<=minIndex;i++){
           //            if(labelList[idx[i]]!=p){
           //                error+=W[idx[i]];

           //            }
           //        }
           //        for(int i=minIndex;i<sampleNum;i++){
           //            if(labelList[idx[i]]==p){
           //                error+=W[idx[i]];

           //            }
           //        }
           //        cout<<error<<" ";


       }
       /**********************************************************/
       cout<<"/**********************************************************/"<<endl;
       //寻找最佳分类器
       float minError=10;
       int minIndex;
       for(int i=0;i<featureNum;i++){
           if(errorFeature[i]<minError){
               minError=errorFeature[i];
               minIndex=i;
           }
       }
       cout<<"minError:"<<minError<<endl;
       //构建分类器
       minIndexList.push_back(minIndex);  //最优分类器索引
       pList.push_back(p[minIndex]);  //确定分类器的方向
       thetaList.push_back(thetaFeature[minIndex]);  //分类器的阈值
       aList.push_back(log((1-minError)/minError));  //分类器的系数
       cout<<"minIndex:"<<minIndex<<endl;
       cout<<"p:"<<p[minIndex]<<endl;
       cout<<"theta"<<thetaFeature[minIndex]<<endl;
       cout<<"a:"<<log((1-minError)/minError)<<endl;
       //更新权重
       fff=0;
       for(int i=0;i<sampleNum;i++){
           if(featureArray[minIndex][i]<=thetaFeature[minIndex]){
               if(labelList[i]==p[minIndex]){
                   W[i]=W[i]*(minError/(1-minError));
               }
           }else{
               if(labelList[i]!=p[minIndex]){
                   W[i]=W[i]*(minError/(1-minError));
               }
           }
           fff+=W[i];
       }
       //权重归一化
       for(int i=0;i<sampleNum;i++){
           W[i]=W[i]/fff;
       }

       TPlus=0;TMinus=0;
       for(int i=0;i<sampleNum;i++){
           if(labelList[i]==1){
               TPlus+=W[i];
           }else{
               TMinus+=W[i];
           }

       }
       //设置阈值
       if(minError<0.000001){
           break;
       }
   }

}
//void adaboostSon(float errorSample[sampleNum],
//                 float errorFeature[featureNum],
//                 float thetaSample[sampleNum],
//                 float thetaFeature[featureNum],
//                 int p[featureNum],vector<int> idx){
//    //排序（记录索引）
//    iota(idx.begin(),idx.end(),0);
//    sort(idx.begin(),idx.end(), [&](int i1, int i2){return featureArray[k][i1]<featureArray[k][i2];});
//    //寻找本特征的最小误差分类器
//    float minError=10;int minIndex;
//    for(int i=0;i<sampleNum;i++){
//        float SPlus=0,SMinus=0;

//        for(int j=0;j<=i;j++){
//            if(labelList[idx[j]]==1){
//                SPlus+=W[idx[j]];
//            }else{
//                SMinus+=W[idx[j]];
//            }
//        }
//        float e1=SPlus+(TMinus-SMinus);
//        float e2=SMinus+(TPlus-SPlus);

//        errorSample[i]=e1>e2?e2:e1;
//        thetaSample[i]=featureArray[k][i];
//        //找最小误差的划分点
//        if(errorSample[i]<minError){
//            minError=errorSample[i];
//            minIndex=i;
//        }
//    }
//    //            cout<<minError<<" ";

//    thetaFeature[k]=thetaSample[minIndex];
//    errorFeature[k]=minError;

//    //确定分类器的方向
//    int pp=0;
//    for(int i=0;i<=minIndex;i++){
//        if(labelList[idx[i]]==1){
//            pp++;
//        }
//    }
//    p[k]=pp/float(minIndex)>0.5?1:-1;
//    //            cout<<thetaFeature[k]<<" ";

//    //计算分类器误差(经验证错误率等于minError)
//    //        float error=0;
//    //        for(int i=0;i<=minIndex;i++){
//    //            if(labelList[idx[i]]!=p){
//    //                error+=W[idx[i]];

//    //            }
//    //        }
//    //        for(int i=minIndex;i<sampleNum;i++){
//    //            if(labelList[idx[i]]==p){
//    //                error+=W[idx[i]];

//    //            }
//    //        }
//    //        cout<<error<<" ";

//}


void quickSort(int left,int right,int k){
   if(left+2<=right){
       int pivot=median3(left,right,k);
       int i=left,j=right-1;
       while(true){
           while(featureArray[++i][k]<pivot){};
           while(featureArray[--j][k]>pivot){};
           if(i<j){
               swapReference(i,j);
           }else{
               break;
           }
       }
       swapReference(i,right-1);
       quickSort(left,i-1,k);
       quickSort(i+1,right,k);

   }else{
       insertionSort(left,right,k);
   }
}

int median3(int left,int right,int k){
   int center=(left+right)/2;
   if(featureArray[center][k]<featureArray[left][k]){
       swapReference(center,left);
   }
   if(featureArray[left][k]>featureArray[right][k]){
       swapReference(left,right);
   }
   if(featureArray[right][k]<featureArray[center][k]){
       swapReference(right,center);
   }
   swapReference(center,right-1);
   return featureArray[right-1][k];
}

void swapReference(int m,int n){
   vector<int> temp=featureArray[m];
   featureArray[m]=featureArray[n];
   featureArray[n]=temp;
   float temp2=W[m];
   W[m]=W[n];
   W[n]=temp2;
}

void insertionSort( int left, int right,int k){
   for (int i = left + 1; i <= right; i++) {
       int t = featureArray[i][k];
       for (int j = i; j > left; j--) {
           if (featureArray[j - 1][k]>t) {
               featureArray[j][k] = featureArray[j - 1][k];
           } else {
               featureArray[j][k] = t;
           }
       }
   }
}











