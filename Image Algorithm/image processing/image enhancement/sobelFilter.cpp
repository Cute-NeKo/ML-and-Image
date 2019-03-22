//#include <opencv2/opencv.hpp>
//#include<iostream>

//using namespace std;
//using namespace cv;
//Mat soble(Mat img);

//int main()
//{
//    Mat img=imread("H://hhh6.jpg");
//    //    imshow("image",img);

//    Mat img2=soble(img);
//    imshow("image",img2);
//    waitKey(0);
//    return 0;
//}


//Mat soble(Mat img){
//    Mat changeImg(img.rows,img.cols,CV_8U);

//    for(int i=0;i<img.rows;i++){
//        for(int j=0;j<img.cols;j++){
//            int x=i;
//            int y=j;
//            if(x<=0){
//                x=1;
//            }
//            if(x>=img.rows-1){
//                x=img.rows-2;
//            }
//            if(y<=0){
//                y=1;
//            }
//            if(y>=img.cols-1){
//                y=img.cols-2;
//            }

//            int filterX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};  //三阶反距离平方权差分法
//            int filterY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};

//            double fx=0.0,fy=0.0;
//            for(int m=-1;m<2;m++){
//                for(int n=-1;n<2;n++){
//                    fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[0];
//                    fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[0];
//                }
//            }
//            //            fx=fx/8.0;
//            //            fy=fy/8.0;
//            double f=fabs(fx)+fabs(fy);

//            changeImg.at<uchar>(i,j)=(uchar)f;

//        }
//    }
//    return changeImg;

//}




