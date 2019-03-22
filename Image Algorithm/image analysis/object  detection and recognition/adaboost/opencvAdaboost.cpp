
//#include <opencv2\opencv.hpp>
//#include <iostream>
//using namespace std;
//using namespace cv;


////F:/opencv/install/etc/haarcascades/haarcascade_frontalface_alt2.xml

//CascadeClassifier faceCascade;

//int main()
//{
//    faceCascade.load("F:/opencv/install/etc/haarcascades/haarcascade_frontalface_alt2.xml");   //加载分类器，注意文件路径

//    Mat img = imread("h://1520504556299.bmp");
//    Mat imgGray;
//    vector<Rect> faces;

//    if(img.empty())
//    {
//        return 1;
//    }

//    if(img.channels() ==3)
//    {
//        cvtColor(img, imgGray, CV_RGB2GRAY);
//    }
//    else
//    {
//        imgGray = img;
//    }

//    faceCascade.detectMultiScale(imgGray, faces, 1.2, 6, 0, Size(0, 0));   //检测人脸
//    cout<<faces.size()<<endl;
//    if(faces.size()>0)
//    {
//        for(int i =0; i<faces.size(); i++)
//        {
//            /*rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
//                      Scalar(0, 255, 0), 1, 8); */   //框出人脸位置
//        }

//        rectangle(img, Point(396, 46), Point(440, 90),
//                  Scalar(0, 255, 0), 1, 8);    //框出人脸位置
//        rectangle(img, Point(226, 92), Point(264, 129),
//                  Scalar(0, 255, 0), 1, 8);    //框出人脸位置
//        rectangle(img, Point(112, 54), Point(150, 92),
//                  Scalar(0, 255, 0), 1, 8);    //框出人脸位置
//        rectangle(img, Point(300, 92), Point(337, 129),
//                  Scalar(0, 255, 0), 1, 8);    //框出人脸位置
//        rectangle(img, Point(342, 67), Point(379, 104),
//                  Scalar(0, 255, 0), 1, 8);    //框出人脸位置
//        rectangle(img, Point(264, 57), Point(301, 95),
//                  Scalar(0, 255, 0), 1, 8);    //框出人脸位置
//        rectangle(img, Point(184, 56), Point(221, 93),
//                  Scalar(0, 255, 0), 1, 8);    //框出人脸位置
//        rectangle(img, Point(150, 89), Point(187,126),
//                  Scalar(0, 255, 0), 1, 8);    //框出人脸位置
//    }
//    imshow("FacesOfPrettyGirl", img);

//    waitKey(0);
//    return 0;
//}
