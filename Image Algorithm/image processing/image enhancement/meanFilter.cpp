//#include <opencv2/opencv.hpp>
//#include<iostream>

//using namespace std;
//using namespace cv;
//Mat meanFilter(Mat img);

//int main()
//{
//    Mat img=imread("H://hhh6.jpg");
////    imshow("image",img);

//    Mat img2=meanFilter(img);
//    imshow("image",img2);
//    waitKey(0);
//    return 0;
//}


//Mat meanFilter(Mat img){
//    Mat changeImg(img.rows,img.cols,CV_8UC3);

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

//            double filter[3][3]={{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}};

//            double b=0,g=0,r=0;
//            for(int m=-1;m<2;m++){
//                for(int n=-1;n<2;n++){
//                    b+=img.at<Vec3b>(x+m,y+n)[0]*filter[m+1][n+1];
//                    g+=img.at<Vec3b>(x+m,y+n)[1]*filter[m+1][n+1];
//                    r+=img.at<Vec3b>(x+m,y+n)[2]*filter[m+1][n+1];
//                }
//            }

//            Vec3b vecf={(uchar)b,(uchar)g,(uchar)r};
//            changeImg.at<Vec3b>(i,j)=vecf;

//        }
//    }
//    return changeImg;
//}




