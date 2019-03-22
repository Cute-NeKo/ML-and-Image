#include <opencv2/opencv.hpp>
#include<iostream>
#include <QApplication>
#include <QChartView>
#include <QLineSeries>
#include <QValueAxis>
#include<fstream>

QT_CHARTS_USE_NAMESPACE
using namespace std;
using namespace cv;
Mat getRectFilter(int r,int c,int angle);
Mat getRectFilter2(int r,int c,int angle);
void showChart(double**data,int rows,QString title);
Mat getMeanSquareImg(Mat img,Mat rectFilter,Mat sumImg);
void showHist(Mat img);
Mat getEntropy(Mat img,Mat rectFilter);
Mat getSimilarNum(Mat img,Mat rectFilter);
void gaborFilter(Mat img,int DirectionNum);
Mat gaborKernel(int ks, double sig, double th, double lm, double ps,double gama);
Mat gaborSmooth(Mat gaborImg);

int FeatureNum=4;  //特征向量维度
Mat img_type;
Mat *gaborImg;
Mat *gaborImg2;
Mat *gaborImg3;

//均方差与均值
int maasin(int argc, char *argv[]){
   QApplication a(argc, argv);

   int DirectionNum=12;

   Mat imgRGB=imread("h:\\city7.png");
   Mat img=imread("h:\\city7.png",IMREAD_GRAYSCALE);
   medianBlur(img,img,3);

   Mat *meanImg=new Mat[DirectionNum];
   Mat *meanSquareImg=new Mat[DirectionNum];


   //    计算矩形模板均方差后的图像
   double step=15;
   double angle=0;
   for(int i=0;i<DirectionNum;i++){
       Mat sumImg,borderPlusImg;
       Mat rectFilter=getRectFilter(5,20,angle);
       //        cout<<rectFilter<<endl;

       filter2D(img, sumImg, CV_32F, rectFilter);
       sumImg/=100;
       normalize(sumImg,meanImg[i],0,1,CV_MINMAX);

       //        imshow("resultImg:"+to_string(i),resultImg[i]);
       normalize(getMeanSquareImg(img,rectFilter,sumImg),meanSquareImg[i],0,1,CV_MINMAX);

       angle+=step;
       cout<<"mean square:"<<i+1<<endl;
   }

   //        for(int i=0;i<DirectionNum;i++){
   //            imshow("mean sqare:"+to_string(step*i),meanSquareImg[i]);
   //        }

   //求最小方向
   Mat minDirectMat(img.rows,img.cols,CV_8U);
   Mat minDirectImg(img.rows,img.cols,CV_32F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           uchar minDirect=0;
           for(int m=0;m<DirectionNum;m++){
               if(meanSquareImg[m].at<float>(i,j)<meanSquareImg[minDirect].at<float>(i,j)){
                   minDirect=m;
               }
           }
           minDirectMat.at<uchar>(i,j)=minDirect;
           minDirectImg.at<float>(i,j)=meanSquareImg[minDirect].at<float>(i,j);
       }
   }
   normalize(minDirectImg,minDirectImg,0,1,CV_MINMAX);
   imshow("min",minDirectImg);

   //    Mat des;
   //    normalize(minDirectImg,des,0,255,CV_MINMAX);
   //    imwrite("C:\\Users\\uygug\\Desktop\\hhh.png",des);


   //道路方向图
   Mat minDirectColor(img.rows,img.cols,CV_8UC3);
   int step22=255/DirectionNum;
   int kkk=0;
   Vec3b* color=new Vec3b[DirectionNum];
   for(int i=0;i<DirectionNum;i++){

       color[i]={uchar(step22*i),uchar(step22*i),uchar(step22*i)};
       //        //        cout<<color[i]<<" ";
       if(i==DirectionNum-1){
           color[i]={0,0,0};
       }
   }
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int d=minDirectMat.at<uchar>(i,j);
           Vec3b ccc=color[d];
           minDirectColor.at<Vec3b>(i,j)=ccc;
       }
   }
   imshow("min dir color",minDirectColor);


   //gabor滤波
   gaborFilter(img,DirectionNum);

   Mat minGaborImg(img.rows,img.cols,CV_32F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           minGaborImg.at<float>(i,j)=gaborImg[minDirectMat.at<uchar>(i,j)].at<float>(i,j);
       }
   }
   normalize(minGaborImg,minGaborImg,0,1,CV_MINMAX);
   imshow("minGaborImg",minGaborImg);


   //显示折线图
   double** data=new double*[DirectionNum];
   for(int i=0;i<DirectionNum;i++){
       data[i]=new double[2];
       data[i][0]=step*i;
       data[i][1]=meanSquareImg[i].at<float>(150,435);
   }

   //    double** data2=new double*[DirectionNum];
   //    for(int i=0;i<DirectionNum;i++){
   //        data2[i]=new double[2];
   //        data2[i][0]=step*i;
   //        data2[i][1]=meanImg[i].at<float>(150,435);
   //    }

   showChart(data,DirectionNum,QString("角度纹理均方差特征"));
   //    showChart(data2,DirectionNum,QString("角度纹理均值"));


   //    circle(img, Point(300, 52), 10,Scalar(255, 0, 0));
   //    circle(img, Point(437, 160), 6,Scalar(255, 0, 0));

   //提取一条道路上的数值
   ofstream file("h:\\gabor\\zzzz.txt");
   int aa=435,bb=150;
   for(int i=0;i<30;i++){
       circle(imgRGB, Point(aa, bb), 4,Scalar(255, 0, 0));
       bb+=10;
       if(i%2==0){
           aa+=1;
       }
       file<<minDirectImg.at<float>(bb,aa)<< "\n";
       cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minDirectImg.at<float>(bb,aa)<<endl;
       //            cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minGaborImg.at<float>(bb,aa)<<endl;
   }
   file.close();

   //    int aa=100,bb=85;
   //    for(int i=0;i<30;i++){
   //        circle(img, Point(aa, bb), 6,Scalar(255, 0, 0));
   //        aa+=10;
   //        if(i%2==0){
   //            bb+=1;
   //        }

   //        cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minDirectImg.at<float>(bb,aa)<<endl;
   //        cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minGaborImg.at<float>(bb,aa)<<endl;
   //    }

   //均方差直方图显示
   showHist(minDirectImg);

   imshow("img",imgRGB);




   //    Mat classifyImg=FCM_Road(img,5,minDirectMat,meanSquareImg,gaborImg,gaborImg2,gaborImg3);

   //    imshow("classify",classifyImg);
   //    for(int i=0;i<5;i++){
   //        Mat img3(img.rows,img.cols,CV_8UC3);
   //        for(int m=0;m<img.rows;m++){
   //            for(int n=0;n<img.cols;n++){
   //                if(img_type.at<uchar>(m,n)==i){
   //                    Vec3b vvv={255,255,255};
   //                    img3.at<Vec3b>(m,n)=vvv;
   //                }
   //            }
   //        }
   //        imshow("classify img"+to_string(i+4),img3);
   //    }

   waitKey(0);
   return a.exec();
}

//相似数目特征
int main(int argc, char *argv[]){
   QApplication a(argc, argv);

   int DirectionNum=12;

   Mat imgRGB=imread("h:\\city11.png");
   Mat img=imread("h:\\city11.png",IMREAD_GRAYSCALE);
   medianBlur(img,img,3);

   Mat *meanImg=new Mat[DirectionNum];
   Mat *meanSquareImg=new Mat[DirectionNum];


   //    计算矩形模板相似数目特征后的图像
   double step=15;
   double angle=0;
   for(int i=0;i<DirectionNum;i++){
       Mat rectFilter=getRectFilter(5,20,angle);
       //        cout<<rectFilter<<endl;

       //        imshow("resultImg:"+to_string(i),resultImg[i]);
       normalize(getSimilarNum(img,rectFilter),meanSquareImg[i],0,1,CV_MINMAX);

       angle+=step;
       cout<<"mean square:"<<i+1<<endl;
   }

   //        for(int i=0;i<DirectionNum;i++){
   //            imshow("mean sqare:"+to_string(step*i),meanSquareImg[i]);
   //        }

   //求最小方向
   Mat minDirectMat(img.rows,img.cols,CV_8U);
   Mat minDirectImg(img.rows,img.cols,CV_32F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           uchar minDirect=0;
           for(int m=0;m<DirectionNum;m++){
               if(meanSquareImg[m].at<float>(i,j)<meanSquareImg[minDirect].at<float>(i,j)){
                   minDirect=m;
               }
           }
           minDirectMat.at<uchar>(i,j)=minDirect;
           minDirectImg.at<float>(i,j)=meanSquareImg[minDirect].at<float>(i,j);
       }
   }
   normalize(minDirectImg,minDirectImg,0,1,CV_MINMAX);
   imshow("min",minDirectImg);

   //    Mat des;
   //    normalize(minDirectImg,des,0,255,CV_MINMAX);
   //    imwrite("C:\\Users\\uygug\\Desktop\\hhh.png",des);


   //道路方向图
   Mat minDirectColor(img.rows,img.cols,CV_8UC3);
   int step22=255/DirectionNum;
   int kkk=0;
   Vec3b* color=new Vec3b[DirectionNum];
   for(int i=0;i<DirectionNum;i++){

       color[i]={uchar(step22*i),uchar(step22*i),uchar(step22*i)};
       //        //        cout<<color[i]<<" ";
       if(i==DirectionNum-1){
           color[i]={0,0,0};
       }
   }
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int d=minDirectMat.at<uchar>(i,j);
           Vec3b ccc=color[d];
           minDirectColor.at<Vec3b>(i,j)=ccc;
       }
   }
   imshow("min dir color",minDirectColor);


   //gabor滤波
   gaborFilter(img,DirectionNum);

   Mat minGaborImg(img.rows,img.cols,CV_32F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           minGaborImg.at<float>(i,j)=gaborImg[minDirectMat.at<uchar>(i,j)].at<float>(i,j);
       }
   }
   normalize(minGaborImg,minGaborImg,0,1,CV_MINMAX);
   imshow("minGaborImg",minGaborImg);


   //显示折线图
   double** data=new double*[DirectionNum];
   for(int i=0;i<DirectionNum;i++){
       data[i]=new double[2];
       data[i][0]=step*i;
       data[i][1]=meanSquareImg[i].at<float>(150,435);
   }

   //    double** data2=new double*[DirectionNum];
   //    for(int i=0;i<DirectionNum;i++){
   //        data2[i]=new double[2];
   //        data2[i][0]=step*i;
   //        data2[i][1]=meanImg[i].at<float>(160,437);
   //    }

   showChart(data,DirectionNum,QString("角度纹理均方差"));
   //    showChart(data2,DirectionNum,QString("角度纹理均值"));


   //    circle(img, Point(300, 52), 10,Scalar(255, 0, 0));
   //    circle(img, Point(437, 160), 6,Scalar(255, 0, 0));

   //提取一条道路上的数值
   ofstream file("h:\\gabor\\zzzz.txt");
   int aa=435,bb=150;
   for(int i=0;i<30;i++){
       circle(imgRGB, Point(aa, bb), 4,Scalar(255, 255, 255));
       bb+=10;
       if(i%2==0){
           aa+=1;
       }
       file<<minDirectImg.at<float>(bb,aa)<<"\n";
       cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minDirectImg.at<float>(bb,aa)<<endl;
       //            cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minGaborImg.at<float>(bb,aa)<<endl;
   }
   file.close();


   //    ofstream file("h:\\gabor\\zzzz.txt");
   //    int aa=100,bb=85;
   //    for(int i=0;i<30;i++){
   //        circle(img, Point(aa, bb), 6,Scalar(255, 0, 0));
   //        aa+=10;
   //        if(i%2==0){
   //            bb+=1;
   //        }
   //        file<<minDirectImg.at<float>(bb,aa)<<"\n";
   //        cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minDirectImg.at<float>(bb,aa)<<endl;
   //    }
   //    file.close();

   //均方差直方图显示
   showHist(minDirectImg);

   imshow("img",imgRGB);




   Mat classifyImg=FCM_Road(img,5,minDirectMat,meanSquareImg,gaborImg,gaborImg2,gaborImg3);

   imshow("classify",classifyImg);
   for(int i=0;i<5;i++){
       Mat img3(img.rows,img.cols,CV_8UC3);
       for(int m=0;m<img.rows;m++){
           for(int n=0;n<img.cols;n++){
               if(img_type.at<uchar>(m,n)==i){
                   Vec3b vvv={255,255,255};
                   img3.at<Vec3b>(m,n)=vvv;
               }
           }
       }
       imshow("classify img"+to_string(i+4),img3);
   }

   waitKey(0);
   return a.exec();
}

//haar特征
int m2ain(int argc, char *argv[]){
   QApplication a(argc, argv);

   int DirectionNum=13;

   Mat img=imread("h:\\city7.png",IMREAD_GRAYSCALE);

   Mat *haarImg=new Mat[DirectionNum];


   //    计算矩形模板均方差后的图像
   double step=15;
   double angle=0;
   for(int i=0;i<DirectionNum;i++){
       Mat sumImg;
       Mat rectFilter=getRectFilter2(5,20,angle);
       //        cout<<rectFilter<<endl;

       filter2D(img, sumImg, CV_32F, rectFilter);
       haarImg[i]=abs(sumImg);
       //        normalize(haarImg[i],haarImg[i],0,1,CV_MINMAX);


       angle+=step;
       cout<<"haar:"<<i+1<<endl;
   }

   //        for(int i=0;i<DirectionNum;i++){
   //            imshow("mean sqare:"+to_string(step*i),meanSquareImg[i]);
   //        }

   //求最小方向
   Mat minDirectMat(img.rows,img.cols,CV_8U);
   Mat minDirectImg(img.rows,img.cols,CV_32F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           uchar minDirect=0;
           for(int m=0;m<DirectionNum;m++){
               if(haarImg[m].at<float>(i,j)<haarImg[minDirect].at<float>(i,j)){
                   minDirect=m;
               }
           }
           minDirectMat.at<uchar>(i,j)=minDirect;
           minDirectImg.at<float>(i,j)=haarImg[minDirect].at<float>(i,j);
       }
   }
   imshow("min",minDirectImg);

   double** data=new double*[DirectionNum];
   for(int i=0;i<DirectionNum;i++){
       data[i]=new double[2];
       data[i][0]=step*i;
       data[i][1]=haarImg[i].at<float>(180,437);
   }

   //    double** data2=new double*[DirectionNum];
   //    for(int i=0;i<DirectionNum;i++){
   //        data2[i]=new double[2];
   //        data2[i][0]=step*i;
   //        data2[i][1]=meanImg[i].at<float>(160,437);
   //    }

   showChart(data,DirectionNum,QString("角度纹理均方差"));
   //    showChart(data2,DirectionNum,QString("角度纹理均值"));


   //    circle(img, Point(300, 52), 10,Scalar(255, 0, 0));
   //    circle(img, Point(437, 160), 6,Scalar(255, 0, 0));

   //提取一条道路上的数值
   int aa=437,bb=160;
   for(int i=0;i<30;i++){
       circle(img, Point(aa, bb), 6,Scalar(255, 0, 0));
       bb+=10;
       if(i%2==0){
           aa+=1;
       }
       cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minDirectImg.at<float>(bb,aa)<<endl;
   }

   //    int aa=100,bb=85;
   //    for(int i=0;i<30;i++){
   //        circle(img, Point(aa, bb), 6,Scalar(255, 0, 0));
   //        aa+=10;
   //        if(i%2==0){
   //            bb+=1;
   //        }

   //        cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minDirectImg.at<float>(bb,aa)<<endl;
   //    }

   //均方差直方图显示
   //    showHist(minDirectImg);

   //阈值分割
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){

           if(minDirectImg.at<float>(i,j)>300){
               minDirectImg.at<float>(i,j)=0;
           }else{
               minDirectImg.at<float>(i,j)=1;
           }
       }
   }
   imshow("min2",minDirectImg);

   imshow("img",img);


   waitKey(0);
   return a.exec();
}

//熵
int ma22in(int argc, char *argv[]){
   QApplication a(argc, argv);

   int DirectionNum=12;

   Mat img=imread("h:\\city7.png",IMREAD_GRAYSCALE);
   medianBlur(img,img,3);

   Mat *meanImg=new Mat[DirectionNum];
   Mat *meanSquareImg=new Mat[DirectionNum];


   //    计算矩形模板均方差后的图像
   double step=15;
   double angle=0;
   for(int i=0;i<DirectionNum;i++){
       Mat sumImg,borderPlusImg;
       Mat rectFilter=getRectFilter(5,20,angle);
       //        cout<<rectFilter<<endl;

       //        imshow("resultImg:"+to_string(i),resultImg[i]);
       normalize(getEntropy(img,rectFilter),meanSquareImg[i],0,1,CV_MINMAX);

       angle+=step;
       cout<<"mean square:"<<i+1<<endl;
   }

   //        for(int i=0;i<DirectionNum;i++){
   //            imshow("mean sqare:"+to_string(step*i),meanSquareImg[i]);
   //        }

   //求最小方向
   Mat minDirectMat(img.rows,img.cols,CV_8U);
   Mat minDirectImg(img.rows,img.cols,CV_32F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           uchar minDirect=0;
           for(int m=0;m<DirectionNum;m++){
               if(meanSquareImg[m].at<float>(i,j)<meanSquareImg[minDirect].at<float>(i,j)){
                   minDirect=m;
               }
           }
           minDirectMat.at<uchar>(i,j)=minDirect;
           minDirectImg.at<float>(i,j)=meanSquareImg[minDirect].at<float>(i,j);
       }
   }
   imshow("min",minDirectImg);
   Mat des;
   normalize(minDirectImg,des,0,255,CV_MINMAX);
   imwrite("C:\\Users\\uygug\\Desktop\\hhh.png",des);


   //道路方向图
   Mat minDirectColor(img.rows,img.cols,CV_8UC3);
   int step22=255/DirectionNum;
   int kkk=0;
   Vec3b* color=new Vec3b[DirectionNum];
   for(int i=0;i<DirectionNum;i++){
       color[i]={uchar(step22*i),uchar(step22*i*(rand()%2)),uchar(step22*i*(rand()%2))};
       //        cout<<color[i]<<" ";
   }
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int d=minDirectMat.at<uchar>(i,j);
           Vec3b ccc=color[d];
           minDirectColor.at<Vec3b>(i,j)=ccc;
       }
   }
   imshow("min dir color",minDirectColor);




   //gabor滤波
   gaborFilter(img,DirectionNum);

   Mat minGaborImg(img.rows,img.cols,CV_32F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           minGaborImg.at<float>(i,j)=gaborImg[minDirectMat.at<uchar>(i,j)].at<float>(i,j);
       }
   }
   imshow("minGaborImg",minGaborImg);


   //显示折线图
   double** data=new double*[DirectionNum];
   for(int i=0;i<DirectionNum;i++){
       data[i]=new double[2];
       data[i][0]=step*i;
       data[i][1]=meanSquareImg[i].at<float>(180,437);
   }

   //    double** data2=new double*[DirectionNum];
   //    for(int i=0;i<DirectionNum;i++){
   //        data2[i]=new double[2];
   //        data2[i][0]=step*i;
   //        data2[i][1]=meanImg[i].at<float>(160,437);
   //    }

   showChart(data,DirectionNum,QString("角度纹理均方差"));
   //    showChart(data2,DirectionNum,QString("角度纹理均值"));


   //    circle(img, Point(300, 52), 10,Scalar(255, 0, 0));
   //    circle(img, Point(437, 160), 6,Scalar(255, 0, 0));

   //提取一条道路上的数值
   //    int aa=437,bb=160;
   //    for(int i=0;i<30;i++){
   //        circle(img, Point(aa, bb), 6,Scalar(255, 0, 0));
   //        bb+=10;
   //        if(i%2==0){
   //            aa+=1;
   //        }
   //        cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minDirectImg.at<float>(bb,aa)<<endl;
   //        cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minGaborImg.at<float>(bb,aa)<<endl;
   //    }

   int aa=100,bb=85;
   for(int i=0;i<30;i++){
       circle(img, Point(aa, bb), 6,Scalar(255, 0, 0));
       aa+=10;
       if(i%2==0){
           bb+=1;
       }

       cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minDirectImg.at<float>(bb,aa)<<endl;
       cout<<(int)minDirectMat.at<uchar>(bb,aa)<<":"<<minGaborImg.at<float>(bb,aa)<<endl;
   }

   //均方差直方图显示
   showHist(minDirectImg);

   imshow("img",img);




   Mat classifyImg=FCM_Road(img,4,minDirectMat,meanSquareImg,gaborImg,gaborImg2,gaborImg3);

   imshow("classify",classifyImg);
   for(int i=0;i<4;i++){
       Mat img3(img.rows,img.cols,CV_8UC3);
       for(int m=0;m<img.rows;m++){
           for(int n=0;n<img.cols;n++){
               if(img_type.at<uchar>(m,n)==i){
                   Vec3b vvv={255,255,255};
                   img3.at<Vec3b>(m,n)=vvv;
               }
           }
       }
       imshow("classify img"+to_string(i+4),img3);
   }

   waitKey(0);
   return a.exec();
}

//折线图显示
void showChart(double**data,int rows,QString title){
   // 构建 series，作为图表的数据源，为其添加 6 个坐标点
   QLineSeries *series = new QLineSeries();
   for(int i=0;i<rows;i++){
       series->append(data[i][0],data[i][1]);
   }

   // 构建图表
   QChart *chart = new QChart();
   chart->legend()->hide();  // 隐藏图例
   chart->addSeries(series);  // 将 series 添加至图表中
   chart->createDefaultAxes();  // 基于已添加到图表的 series 来创轴
   chart->setTitle(title);  // 设置图表的标题

   //创建QAxis
   QValueAxis *axisX = new QValueAxis;
   axisX->setRange(0, 180);
   axisX->setLabelFormat("%u"); //设置刻度的格式
   axisX->setGridLineVisible(true);
   axisX->setTickCount(13);     //设置多少格
   axisX->setMinorTickCount(1); //设置每格小刻度线的数目
   chart->setAxisX(axisX, series);

   // 构建 QChartView，并设置抗锯齿、标题、大小
   QChartView *chartView = new QChartView(chart);
   chartView->setRenderHint(QPainter::Antialiasing);
   chartView->setWindowTitle(title);
   chartView->resize(800, 300);
   chartView->show();
}

//生成矩形模板
Mat getRectFilter(int r,int c,int angle){
   int size=r>c?r:c;
   int borderAll=6;
   if(size%2==0){
       size+=(borderAll+1);
   }else{
       size+=borderAll;
   }
   Mat baseImg(size,size,CV_32F,Scalar(0));
   int topR=(size-r)/2,topC=(size-c)/2;
   int buttonR=topR+r,buttonC=topC+c;
   for(int i=topR;i<buttonR;i++){
       for(int j=topC;j<buttonC;j++){
           baseImg.at<float>(i,j)=1;
       }
   }

   int centerR=size/2;
   int centerC=size/2;
   Mat rotationMat=getRotationMatrix2D(Point2f(centerR,centerC),angle,1.0);
   Mat img2;
   warpAffine(baseImg,img2,rotationMat,baseImg.size());

   //矩形模板二值化
   for(int i=0;i<img2.rows;i++){
       for(int j=0;j<img2.cols;j++){
           if(img2.at<float>(i,j)>0.5){
               img2.at<float>(i,j)=1;
           }else{
               img2.at<float>(i,j)=0;
           }
       }
   }
   return img2;

}

//生成矩形模板
Mat getRectFilter2(int r,int c,int angle){
   int size=r>c?r:c;
   int borderAll=6;
   if(size%2==0){
       size+=(borderAll+1);
   }else{
       size+=borderAll;
   }
   Mat baseImg(size,size,CV_32F,Scalar(0));
   int topR=(size-r)/2,topC=(size-c)/2;
   int buttonR=topR+r,buttonC=topC+c;
   for(int i=topR;i<=buttonR;i++){
       int num=0;
       for(int j=topC;j<buttonC;j++){
           num++;
           if(num<=10){
               baseImg.at<float>(i,j)=1;
           }else{
               baseImg.at<float>(i,j)=-1;
           }

       }
   }

   int centerR=size/2;
   int centerC=size/2;
   Mat rotationMat=getRotationMatrix2D(Point2f(centerR,centerC),angle,1.0);
   Mat img2;
   warpAffine(baseImg,img2,rotationMat,baseImg.size());

   //矩形模板二值化
   for(int i=0;i<img2.rows;i++){
       for(int j=0;j<img2.cols;j++){
           if(img2.at<float>(i,j)>0.5){
               img2.at<float>(i,j)=1;
           }else if(img2.at<float>(i,j)<-0.5){
               img2.at<float>(i,j)=-1;
           }else{
               img2.at<float>(i,j)=0;
           }
       }
   }
   return img2;

}

//进行均方差滤波
Mat getMeanSquareImg(Mat img,Mat rectFilter,Mat sumImg){
   int size=rectFilter.cols;
   Mat borderPlusImg;
   copyMakeBorder(img,borderPlusImg,rectFilter.rows/2,rectFilter.rows/2,rectFilter.rows/2,rectFilter.rows/2, BORDER_REFLECT_101);
   Mat resultImg(img.rows,img.cols,CV_32F);
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           float square=0;
           int num=0;
           for(int m=0;m<size;m++){
               for(int n=0;n<size;n++){
                   if(rectFilter.at<float>(m,n)>0.5){
                       //计算均方差
                       num++;
                       square+=pow((borderPlusImg.at<uchar>(i+m,j+n)-sumImg.at<float>(i,j)),2);
                   }

               }
           }

           resultImg.at<float>(i,j)=sqrt(square/num);
       }
   }
   //    return resultImg;

   Mat changeResultImg(img.rows,img.cols,CV_32F);
   //均方差调整，使道路宽度正常
   for(int i=0;i<resultImg.rows;i++){
       for(int j=0;j<resultImg.cols;j++){
           int x=i;
           int y=j;
           if(x<=0){
               x=1;
           }
           if(x>=resultImg.rows-1){
               x=resultImg.rows-2;
           }
           if(y<=0){
               y=1;
           }
           if(y>=resultImg.cols-1){
               y=resultImg.cols-2;
           }


           float minSquare=10000;
           for(int m=-1;m<2;m++){
               for(int n=-1;n<2;n++){
                   float bbb=resultImg.at<float>(x+m,y+n);
                   if(bbb<minSquare){
                       minSquare=bbb;
                   }
               }
           }

           changeResultImg.at<float>(i,j)=minSquare;

       }
   }


   return changeResultImg;

}

//图像熵计算
Mat getEntropy(Mat img,Mat rectFilter){
   int size=rectFilter.cols;
   Mat borderPlusImg;
   copyMakeBorder(img,borderPlusImg,rectFilter.rows/2,rectFilter.rows/2,rectFilter.rows/2,rectFilter.rows/2, BORDER_REFLECT_101);
   Mat resultImg(img.rows,img.cols,CV_32F);

   double *hist=new double[256];

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           for(int z=0;z<256;z++){
               hist[z]=0;
           }
           float entropy=0;
           int num=0;

           for(int m=0;m<size;m++){
               for(int n=0;n<size;n++){
                   if(rectFilter.at<float>(m,n)>0.5){
                       //计算直方图
                       num++;
                       hist[borderPlusImg.at<uchar>(i+m,j+n)]++;
                   }

               }
           }

           //计算熵
           for(int z=0;z<256;z++){
               if(hist[z]!=0){
                   float p=hist[z]/float(num);
                   entropy+=-(p*log(p));
               }
           }

           resultImg.at<float>(i,j)=entropy;
       }
   }
   delete[] hist;
   //    return resultImg;
   Mat changeResultImg(img.rows,img.cols,CV_32F);
   //均方差调整，使道路宽度正常
   for(int i=0;i<resultImg.rows;i++){
       for(int j=0;j<resultImg.cols;j++){
           int x=i;
           int y=j;
           if(x<=0){
               x=1;
           }
           if(x>=resultImg.rows-1){
               x=resultImg.rows-2;
           }
           if(y<=0){
               y=1;
           }
           if(y>=resultImg.cols-1){
               y=resultImg.cols-2;
           }


           float minSquare=10000;
           for(int m=-1;m<2;m++){
               for(int n=-1;n<2;n++){
                   float bbb=resultImg.at<float>(x+m,y+n);
                   if(bbb<minSquare){
                       minSquare=bbb;
                   }
               }
           }

           changeResultImg.at<float>(i,j)=minSquare;

       }
   }


   return changeResultImg;

}

//图像矩形模板相似数目计算
Mat getSimilarNum(Mat img,Mat rectFilter){
   int size=rectFilter.cols;
   Mat borderPlusImg;
   copyMakeBorder(img,borderPlusImg,rectFilter.rows/2,rectFilter.rows/2,rectFilter.rows/2,rectFilter.rows/2, BORDER_REFLECT_101);
   Mat resultImg(img.rows,img.cols,CV_32F);

   double *hist=new double[256];

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           float num=0;

           for(int m=0;m<size;m++){
               for(int n=0;n<size;n++){
                   if(rectFilter.at<float>(m,n)>0.5){
                       float aaa=abs(borderPlusImg.at<uchar>(i+m,j+n)-img.at<uchar>(i,j));
                       if(aaa<=10){
                           num++;
                       }
                       else if(aaa<=50){
                           num+=(-(1/40.0)*aaa+(1/40.0*50));
                       }
                   }

               }
           }


           resultImg.at<float>(i,j)=100-num;
       }
   }
   //    return resultImg;

   Mat changeResultImg(img.rows,img.cols,CV_32F);
   //均方差调整，使道路宽度正常
   for(int i=0;i<resultImg.rows;i++){
       for(int j=0;j<resultImg.cols;j++){
           int x=i;
           int y=j;
           if(x<=0){
               x=1;
           }
           if(x>=resultImg.rows-1){
               x=resultImg.rows-2;
           }
           if(y<=0){
               y=1;
           }
           if(y>=resultImg.cols-1){
               y=resultImg.cols-2;
           }


           float minSquare=10000;
           for(int m=-1;m<2;m++){
               for(int n=-1;n<2;n++){
                   float bbb=resultImg.at<float>(x+m,y+n);
                   if(bbb<minSquare){
                       minSquare=bbb;
                   }
               }
           }

           changeResultImg.at<float>(i,j)=minSquare;

       }
   }


   return changeResultImg;
}

//得到直方图
void showHist(Mat img){
   double *hist=new double[250];
   for(int i=0;i<250;i++){
       hist[i]=0;
   }
   double step=1.0/250;
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           float a=img.at<float>(i,j);
           int index=int(a/step);
           hist[index]++;
       }
   }
   int max=0;
   for(int j=0;j<256;j++){
       if(hist[j]>max){
           max=hist[j];
       }
   }

   for(int j=0;j<250;j++){
       hist[j]/=max;
   }

   Mat histPic(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));


   for(int i=0;i<250;i++){
       line(histPic,Point(i,250),Point(i,250-250*hist[i]),Scalar(255,0,0));
   }


   imshow("image",histPic);

}

//生成gabor核
Mat gaborKernel(int ks, double sig, double th, double lm, double ps,double gama)
{
   int hks = (ks-1)/2;
   float theta = -th*CV_PI/180;
   float psi = ps*CV_PI/180;
   float del = 2.0/(ks-1);
   float lmbd = lm;
   float sigma = sig/ks;
   float x_theta;
   float y_theta;
   cv::Mat kernel(ks,ks, CV_32F);
   for (int y=-hks; y<=hks; y++)
   {
       for (int x=-hks; x<=hks; x++)
       {
           x_theta = x*del*cos(theta)+y*del*sin(theta);
           y_theta = -x*del*sin(theta)+y*del*cos(theta);
           kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(gama,2)*pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
       }
   }
   return kernel;
}

//gabor滤波
void gaborFilter(Mat img,int DirectionNum){
   float kernel_size=21;
   float pos_sigma= 10;
   float pos_lm = 80;
   float *pos_ths;
   float pos_psi = 90;
   float pos_gama=4;
   const int FeatureNum=4;


   gaborImg=new Mat[DirectionNum];
   gaborImg2=new Mat[DirectionNum];
   gaborImg3=new Mat[DirectionNum];
   double step=180/DirectionNum;
   double angle=0;
   pos_ths=new float[DirectionNum];
   for(int i=0;i<DirectionNum;i++){
       pos_ths[i]=angle;
       cout<<angle<<endl;
       angle+=step;
   }

   float sig = pos_sigma;
   float lm1 = 0.5+pos_lm/100.0;
   float lm2 = 0.5+(pos_lm-30)/100.0;
   float lm3 = 0.5+(pos_lm-30)/100.0;
   float* ths = pos_ths;
   float ps = pos_psi;
   float gama=pos_gama;
   //12个方向gabor滤波(三次)
   for(int i=0;i<DirectionNum;i++){
       Mat dest;
       //        cv::Mat kernel = gaborKernel(kernel_size, sig, ths[i], 0.5+(pos_lm-10)/100.0, ps,gama);
       cv::Mat kernel = gaborKernel(kernel_size, sig, ths[i], lm1, ps,gama);

       cv::filter2D(img, dest, CV_32F, kernel);

       cv::Mat hhh;
       cv::pow(dest, 2.0, hhh);
       cv::normalize(hhh,hhh,0,255,CV_MINMAX);

       hhh.convertTo(hhh,CV_8U);
       equalizeHist(hhh,hhh);
       hhh.convertTo(hhh,CV_32F);
       cv::normalize(hhh,hhh,0,1,CV_MINMAX);

       //        dest.copyTo(gaborImg[i]);
       gaborImg[i]=gaborSmooth(hhh);


       cv::normalize(gaborImg[i],gaborImg[i],0,1,CV_MINMAX);
   }

   for(int i=0;i<DirectionNum;i++){
       Mat dest;
       //        cv::Mat kernel = gaborKernel(kernel_size, sig, ths[i], 0.5+(pos_lm+20)/100.0, ps,gama);
       cv::Mat kernel = gaborKernel(kernel_size, sig, ths[i], lm2, ps,gama);
       cv::filter2D(img, dest, CV_32F, kernel);
       cv::Mat hhh;
       cv::pow(dest, 2.0, hhh);
       cv::normalize(hhh,hhh,0,255,CV_MINMAX);

       hhh.convertTo(hhh,CV_8U);
       equalizeHist(hhh,hhh);
       hhh.convertTo(hhh,CV_32F);
       cv::normalize(hhh,hhh,0,1,CV_MINMAX);

       //        dest.copyTo(gaborImg[i]);
       gaborImg2[i]=gaborSmooth(hhh);


       cv::normalize(gaborImg2[i],gaborImg2[i],0,1,CV_MINMAX);
   }

   for(int i=0;i<DirectionNum;i++){
       Mat dest;
       cv::Mat kernel = gaborKernel(kernel_size, sig, ths[i], lm3, ps,gama);
       cv::filter2D(img, dest, CV_32F, kernel);
       cv::Mat hhh;
       cv::pow(dest, 2.0, hhh);
       cv::normalize(hhh,hhh,0,255,CV_MINMAX);

       hhh.convertTo(hhh,CV_8U);
       equalizeHist(hhh,hhh);
       hhh.convertTo(hhh,CV_32F);
       cv::normalize(hhh,hhh,0,1,CV_MINMAX);

       //        dest.copyTo(gaborImg[i]);
       gaborImg3[i]=gaborSmooth(hhh);


       cv::normalize(gaborImg3[i],gaborImg3[i],0,1,CV_MINMAX);
   }

}

//gabor滤波结果平滑
Mat gaborSmooth(Mat gaborImg){
   Mat resultImg(gaborImg.rows,gaborImg.cols,CV_32F);
   //均方差调整，使道路宽度正常
   for(int i=0;i<gaborImg.rows;i++){
       for(int j=0;j<gaborImg.cols;j++){
           int x=i;
           int y=j;
           if(x<=0){
               x=1;
           }
           if(x>=gaborImg.rows-1){
               x=gaborImg.rows-2;
           }
           if(y<=0){
               y=1;
           }
           if(y>=gaborImg.cols-1){
               y=gaborImg.cols-2;
           }


           float minGabor=10000;
           for(int m=-1;m<2;m++){
               for(int n=-1;n<2;n++){
                   float bbb=gaborImg.at<float>(x+m,y+n);
                   if(bbb<minGabor){
                       minGabor=bbb;
                   }
               }
           }

           resultImg.at<float>(i,j)=minGabor;

       }
   }
   return resultImg;
}

