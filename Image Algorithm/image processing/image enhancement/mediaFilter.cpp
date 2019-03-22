//#include <opencv2/opencv.hpp>
//#include<iostream>

//using namespace std;
//using namespace cv;

//Mat medianFiltering(Mat img);

//int main()
//{
//    Mat img=imread("H://hhh4.png");
//    imshow("image",img);

//    Mat img2=medianFiltering(img);

//    imshow("image2",img2);
//    waitKey(0);
//    return 0;
//}

//Mat medianFiltering(Mat img){
//    Mat img2(img.rows,img.cols,CV_8UC3);
//    int imgArray[9][3];
//    for(int i=0;i<img.rows;i++){
//        for(int j=0;j<img.cols;j++){

//            int x=i,y=j;
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


//            int k=0;
//            for(int s=-1;s<2;s++){
//                for(int z=-1;z<2;z++){
//                    imgArray[k][0]=img.at<Vec3b>(x+s,y+z)[0];
//                    imgArray[k][1]=img.at<Vec3b>(x+s,y+z)[1];
//                    imgArray[k][2]=img.at<Vec3b>(x+s,y+z)[2];

//                    k++;
//                }
//            }

//            for(int i=0;i<3;i++){
//                for(int j=0;j<8;j++){
//                    int min=j;
//                    for(int z=j;z<9;z++){
//                        if(imgArray[z][i]<imgArray[min][i]){
//                            min=z;
//                        }
//                    }
//                    if(min!=j){
//                        int t=imgArray[min][i];
//                        imgArray[min][i]=imgArray[j][i];
//                        imgArray[j][i]=t;
//                    }
//                }
//            }


//            Vec3b temp={imgArray[4][0],imgArray[4][1],imgArray[4][2]};
//            img2.at<Vec3b>(x,y)=temp;

//        }
//    }


//    return img2;

//}





