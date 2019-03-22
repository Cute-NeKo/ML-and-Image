//#include <opencv2/opencv.hpp>
//#include<iostream>

//using namespace std;
//using namespace cv;

//int main(){
//    //    Mat img = imread("H:\\class888.png");
//    //    imshow("原始图", img);
//    //    Mat out;
//    //    //获取自定义核
//    //    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的

//    //    morphologyEx(img, out, MORPH_CLOSE, element);
//    //    imshow("膨胀操作", out);
//    //    waitKey(0);


//    Mat imag, result;
//    imag = imread("H:\\class888.png",0);    //将读入的彩色图像直接以灰度图像读入
////    namedWindow("原图", 1);
//    imshow("原图", imag);
//    result = imag.clone();
//    //进行二值化处理，选择30，200.0为阈值
//    threshold(imag, result, 30, 255.0, CV_THRESH_BINARY);
////    namedWindow("二值化图像");
//    imshow("二值化图像", result);
//    imwrite("H:\\class8888.png",result);
//    waitKey();
//    return 0;
//}
