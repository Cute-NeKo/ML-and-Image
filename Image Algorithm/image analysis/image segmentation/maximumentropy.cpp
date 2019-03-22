#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
int maximumEntropy(Mat inputImg);

int main(){
    Mat img=imread("H:\\hhh3.jpg");
    maximumEntropy(img);
    waitKey(0);
    return 0;
}

int maximumEntropy(Mat inputImg){
    Mat grayImg;
    cvtColor(inputImg,grayImg,COLOR_BGR2GRAY);
    int nHist[256];
    for(int i=0;i<256;i++){
        nHist[i]=0;
    }
    for(int i=0;i<grayImg.rows;i++){
        for(int j=0;j<grayImg.cols;j++){
            int ss=grayImg.at<uchar>(i,j);
            if(ss<256){
                nHist[ss]++;
            }
        }
    }

    double pHist[256]{0};
    for(int i=0;i<256;i++){
        pHist[i]=nHist[i]/(double(grayImg.rows*grayImg.cols));
    }

    double PHist[256]{0};
    for(int i=0;i<256;i++){
        for(int j=0;j<=i;j++){
            PHist[i]+=pHist[j];
        }
    }
    PHist[255]=1;


    double theta[256]{0};
    for(int i=0;i<256;i++){
        double e1=0,e2=0;
        for(int j=0;j<=i;j++){
            if(pHist[j]/PHist[i]>0.0000001)
                e1-=(pHist[j]/PHist[i])*log(pHist[j]/PHist[i]);

        }
        for(int j=i+1;j<256;j++){
            if(pHist[j]/(1-PHist[i])>0.0000001)
                e2-=(pHist[j]/double(1-PHist[i]))*log(pHist[j]/double(1-PHist[i]));
        }
        theta[i]=e1+e2;
    }



    int thre=0;
    double max=0;
    for(int i=0;i<256;i++){
        if(theta[i]>max){
            max=theta[i];
            thre=i;
        }
    }
    cout<<thre<<endl;

    for(int i=0;i<grayImg.rows;i++){
        for(int j=0;j<grayImg.cols;j++){
            if(grayImg.at<uchar>(i,j)<thre){
                grayImg.at<uchar>(i,j)=0;
            }else{
                grayImg.at<uchar>(i,j)=255;
            }
        }
    }
    imshow("GG",grayImg);


}
