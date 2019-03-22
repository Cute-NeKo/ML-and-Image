//#include<iostream>
//#include<opencv2/opencv.hpp>

//using namespace std;
//using namespace cv;
//int otsu(Mat inputImg);

//int main(){
//    Mat img=imread("H:\\hhh3.jpg");
//    otsu(img);
//    waitKey(0);
//    return 0;
//}

//int otsu(Mat inputImg){
//    Mat grayImg;
//    cvtColor(inputImg,grayImg,COLOR_BGR2GRAY);
//    int nHist[256];
//    for(int i=0;i<256;i++){
//        nHist[i]=0;
//    }
//    for(int i=0;i<grayImg.rows;i++){
//        for(int j=0;j<grayImg.cols;j++){
//            int ss=grayImg.at<uchar>(i,j);
//            if(ss<256){
//                nHist[ss]++;
//            }
//        }
//    }

//    double pHist[256]{0};
//    for(int i=0;i<256;i++){
//        pHist[i]=nHist[i]/(double(grayImg.rows*grayImg.cols));
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
//            ss+=(j*pHist[j]);
//        }
//        MHist[i]=int(ss+0.5);
//    }

//    int MG=MHist[255];
//    long theta[256]{0};
//    for(int i=0;i<256;i++){
//        if((1-PHist[i])<=0){
//            theta[i]=0;
//        }else{
//            double ss=pow(MHist[i]-MG,2)*PHist[i]/(1-PHist[i]);
//            theta[i]=long(ss);
//        }
//    }

//    int thre=0;
//    int max=0;
//    for(int i=0;i<256;i++){
//        if(theta[i]>max){
//            max=theta[i];
//            thre=i;
//        }
//    }
//    cout<<thre<<endl;

//    for(int i=0;i<grayImg.rows;i++){
//        for(int j=0;j<grayImg.cols;j++){
//            if(grayImg.at<uchar>(i,j)<thre){
//                grayImg.at<uchar>(i,j)=0;
//            }else{
//                grayImg.at<uchar>(i,j)=255;
//            }
//        }
//    }
//    imshow("GG",grayImg);


//}
