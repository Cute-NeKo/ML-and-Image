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
double getSim_Road(double *a,double *b);
double RBFfunction(double *a,double *b,double sigma=0.5);
void randCent_Road(double **C,int k);
Mat FCM_Road(Mat img,int k,Mat directImg,Mat *meanSquareImg,Mat *gaborImg,Mat *gaborImg2,Mat *gaborImg3);
Mat FCM_Road_Kernel(Mat img,int k,Mat directImg,Mat *meanSquareImg,Mat *gaborImg,Mat *gaborImg2,Mat *gaborImg3);
Mat FCM_Road_Kernel_Space(Mat img,int k,Mat directImg,Mat *meanSquareImg,Mat *gaborImg,Mat *gaborImg2,Mat *gaborImg3);



int FeatureNum=4;  //特征向量维度
Mat img_type;
Mat *gaborImg;
Mat *gaborImg2;
Mat *gaborImg3;

//均方差与均值
int main(int argc, char *argv[]){
    QApplication a(argc, argv);

    int DirectionNum=12;

    Mat imgRGB=imread("h:\\city7.png");
    imshow("FF",imgRGB);
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




    Mat classifyImg=FCM_Road_Kernel_Space(img,5,minDirectMat,meanSquareImg,gaborImg,gaborImg2,gaborImg3);

    imshow("classify",classifyImg);
    for(int i=0;i<5;i++){
        Mat img3(img.rows,img.cols,CV_8UC3, Scalar::all(0));
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

//相似数目特征
int maiaan(int argc, char *argv[]){
    QApplication a(argc, argv);

    int DirectionNum=12;

    Mat imgRGB=imread("h:\\city7.png");
    Mat img=imread("h:\\city7.png",IMREAD_GRAYSCALE);
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


    imshow("img",imgRGB);




    Mat classifyImg=FCM_Road_Kernel(img,5,minDirectMat,meanSquareImg,gaborImg,gaborImg2,gaborImg3);

    imshow("classify",classifyImg);
    for(int i=0;i<5;i++){
        Mat img3(img.rows,img.cols,CV_8UC3, Scalar::all(0));
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

//相似度计算
double getSim_Road(double *a,double *b){
    double sum=0;
    for(int i=0;i<FeatureNum;i++){
        sum+=(a[i]-b[i])*(a[i]-b[i]);
    }
    return sqrt(sum);
}

//径向基核函数
double RBFfunction(double *a,double *b,double sigma){
    return exp((-pow(getSim_Road(a,b),2))/(2*sigma*sigma));
}

//随机聚类中心
void randCent_Road(double **C,int k){
    srand((unsigned)time(NULL));
    for(int i=0;i<FeatureNum;i++){
        for(int j=0;j<k;j++){
            C[j][i]= rand()%100/(double)101;
        }
    }
}

//注：聚类中心最好为浮点型
Mat FCM_Road(Mat img,int k,Mat directImg,Mat *meanSquareImg,Mat *gaborImg,Mat *gaborImg2,Mat *gaborImg3){
    int m=2,maxStep=20; //m默认为2
    double maxU=1000;
    int size=img.rows*img.cols;
    //    cout<<size<<endl;
    double **U=new double*[k];
    for(int i=0;i<k;i++){
        U[i]=new double[size];
    }
    double **C;
    //构建图像特征矩阵（用于聚类）
    double ***featureMat;
    featureMat=new double**[img.rows];
    for(int i=0;i<img.rows;i++){
        featureMat[i]=new double*[img.cols];
        for(int j=0;j<img.cols;j++){
            featureMat[i][j]=new double[FeatureNum];
            int minDirect=directImg.at<uchar>(i,j);
            featureMat[i][j][0]=meanSquareImg[minDirect].at<float>(i,j);
            featureMat[i][j][1]=gaborImg[minDirect].at<float>(i,j);
            //            featureMat[i][j][2]=gaborImg2[minDirect].at<float>(i,j);
            //            featureMat[i][j][3]=gaborImg3[minDirect].at<float>(i,j);
            featureMat[i][j][2]=gaborImg[minDirect].at<float>(i,j);
            featureMat[i][j][3]=gaborImg[minDirect].at<float>(i,j);
        }
    }
    //特征归一化
    double sumFeature[4]={0,0,0,0};
    double minFeature[4]={1000,1000,1000,1000},maxFeature[4]={0,0,0,0};
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            for(int z=0;z<4;z++){
                if(minFeature[z]>featureMat[i][j][z]){
                    minFeature[z]=featureMat[i][j][z];
                }
                if(maxFeature[z]<featureMat[i][j][z]){
                    maxFeature[z]=featureMat[i][j][z];
                }
            }

        }
    }
    for(int i=0;i<4;i++){
        cout<<maxFeature[i]<<" ";
    }
    cout<<endl;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            for(int z=0;z<4;z++){
                featureMat[i][j][z]=(featureMat[i][j][z]-minFeature[z])/(maxFeature[z]-minFeature[z]);
                sumFeature[z]+=featureMat[i][j][z];
            }
        }
    }

    //初始化隶属度矩阵
    C=new double*[k];
    for(int i=0;i<k;i++){
        C[i]=new double[FeatureNum];
    }

    randCent_Road(C,k);

    for(int i=0;i<k;i++){
        for(int j=0;j<FeatureNum;j++){
            cout<<C[i][j]<<" ";
        }
        cout<<endl;
    }

    for(int i=0;i<k;i++){
        int num=0;
        for(int r=0;r<img.rows;r++){
            for(int c=0;c<img.cols;c++){
                double ss=0;
                for(int j=0;j<k;j++){
                    double ss2=getSim_Road(featureMat[r][c],C[i])/getSim_Road(featureMat[r][c],C[j]);
                    ss+=ss2*ss2;

                }

                U[i][num]=1.0/ss;

                num++;
            }
        }
    }
    //        for(int j=0;j<size;j++){
    //            double ss=0;
    //            for(int i=0;i<k;i++){
    //                cout<<U[i][j]<<" ";
    //            }
    //        }


    cout<<"************************************************************"<<endl;

    int stepk=0;
    while(maxU>0.01&&stepk<15){
        stepk++;
        //计算聚类中心C
        for(int i=0;i<k;i++){
            double *sum1=new double[FeatureNum]();
            double *sum2=new double[FeatureNum]();
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    double uuu=U[i][num]*U[i][num];
                    for(int z=0;z<FeatureNum;z++){
                        sum1[z]+=uuu*featureMat[r][c][z];
                        sum2[z]+=uuu;
                    }

                    num++;
                }
            }

            for(int z=0;z<FeatureNum;z++){
                C[i][z]=sum1[z]/sum2[z];
                cout<<C[i][z]<<" ";
            }
            cout<<endl;
        }


        //计算模糊矩阵U
        maxU=0;
        for(int i=0;i<k;i++){
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    double ss=0;
                    for(int j=0;j<k;j++){
                        double ss2=getSim_Road(featureMat[r][c],C[i])/getSim_Road(featureMat[r][c],C[j]);

                        ss+=ss2*ss2;
                    }
                    double dd=abs(U[i][num]-1.0/ss);
                    if(dd>maxU){
                        maxU=dd;
                    }

                    U[i][num]=1.0/ss;

                    num++;
                }
            }
        }

        //        for(int i=0;i<k;i++){
        //            cout<<U[i][0]<<" ";
        //        }
        //        cout<<endl;


        cout<<"maxU:"<<maxU<<endl;
        cout<<"************************************************************"<<endl;
    }

    cout<<"end"<<endl;

    //    for(int j=0;j<size;j++){
    //        double ss=0;
    //        for(int i=0;i<k;i++){
    //            cout<<U[i][j]<<" ";
    //        }
    //        cout<<endl;

    //    }

    int step=255/k;
    Mat changeImg(img.rows,img.cols,CV_8UC3);
    int kkk=0;
    Vec3b* color=new Vec3b[k];
    for(int i=0;i<k;i++){
        color[i]={uchar(step*i),uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2))};
        cout<<color[i]<<" ";
    }


    img_type=Mat(img.rows,img.cols,CV_8U);
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int type=0;
            double maxK=0;
            for(int m=0;m<k;m++){
                if(U[m][kkk]>maxK){
                    maxK=U[m][kkk];
                    type=m;
                }
            }
            img_type.at<uchar>(i,j)=type;
            Vec3b ccc=color[type];
            changeImg.at<Vec3b>(i,j)=ccc;
            kkk++;
        }
    }
    return changeImg;

}

//注：聚类中心最好为浮点型
Mat FCM_Road_Kernel(Mat img,int k,Mat directImg,Mat *meanSquareImg,Mat *gaborImg,Mat *gaborImg2,Mat *gaborImg3){
    int m=2,maxStep=20; //m默认为2
    double maxU=1000;
    int size=img.rows*img.cols;
    //    cout<<size<<endl;
    double **U=new double*[k];
    for(int i=0;i<k;i++){
        U[i]=new double[size];
    }
    double **C;
    //构建图像特征矩阵（用于聚类）
    double ***featureMat;
    featureMat=new double**[img.rows];
    for(int i=0;i<img.rows;i++){
        featureMat[i]=new double*[img.cols];
        for(int j=0;j<img.cols;j++){
            featureMat[i][j]=new double[FeatureNum];
            int minDirect=directImg.at<uchar>(i,j);
            featureMat[i][j][0]=meanSquareImg[minDirect].at<float>(i,j);
            featureMat[i][j][1]=gaborImg[minDirect].at<float>(i,j);
            //            featureMat[i][j][2]=gaborImg2[minDirect].at<float>(i,j);
            //            featureMat[i][j][3]=gaborImg3[minDirect].at<float>(i,j);
            featureMat[i][j][2]=gaborImg[minDirect].at<float>(i,j);
            featureMat[i][j][3]=meanSquareImg[minDirect].at<float>(i,j);
        }
    }
    //特征归一化
    double sumFeature[4]={0,0,0,0};
    double minFeature[4]={1000,1000,1000,1000},maxFeature[4]={0,0,0,0};
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            for(int z=0;z<4;z++){
                if(minFeature[z]>featureMat[i][j][z]){
                    minFeature[z]=featureMat[i][j][z];
                }
                if(maxFeature[z]<featureMat[i][j][z]){
                    maxFeature[z]=featureMat[i][j][z];
                }
            }

        }
    }
    for(int i=0;i<4;i++){
        cout<<maxFeature[i]<<" ";
    }
    cout<<endl;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            for(int z=0;z<4;z++){
                featureMat[i][j][z]=(featureMat[i][j][z]-minFeature[z])/(maxFeature[z]-minFeature[z]);
                sumFeature[z]+=featureMat[i][j][z];
            }
        }
    }

    //初始化隶属度矩阵
    C=new double*[k];
    for(int i=0;i<k;i++){
        C[i]=new double[FeatureNum];
    }

    randCent_Road(C,k);

    for(int i=0;i<k;i++){
        for(int j=0;j<FeatureNum;j++){
            cout<<C[i][j]<<" ";
        }
        cout<<endl;
    }

    for(int i=0;i<k;i++){
        int num=0;
        for(int r=0;r<img.rows;r++){
            for(int c=0;c<img.cols;c++){
                double ss1=0,ss2=0;
                ss1=1.0/(1-RBFfunction(featureMat[r][c],C[i],150));
                for(int j=0;j<k;j++){
                    double sss=1.0/(1-RBFfunction(featureMat[r][c],C[j],150));
                    ss2+=sss;
                }

                U[i][num]=ss1/ss2;

                num++;
            }
        }
    }
    //    for(int j=0;j<size;j++){
    //        double ss=0;
    //        for(int i=0;i<k;i++){
    //            ss+=U[i][j];
    //        }
    //        cout<<ss<<endl;
    //    }



    cout<<"************************************************************"<<endl;

    int stepk=0;
    while(maxU>0.01&&stepk<15){
        stepk++;
        //计算聚类中心C
        for(int i=0;i<k;i++){
            double *sum1=new double[FeatureNum]();
            double *sum2=new double[FeatureNum]();
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    double uuu=U[i][num]*U[i][num];
                    for(int z=0;z<FeatureNum;z++){
                        sum1[z]+=uuu*featureMat[r][c][z]*RBFfunction(featureMat[r][c],C[i],150);
                        sum2[z]+=uuu*RBFfunction(featureMat[r][c],C[i],150);
                    }

                    num++;
                }
            }

            for(int z=0;z<FeatureNum;z++){
                C[i][z]=sum1[z]/sum2[z];
                cout<<C[i][z]<<" ";
            }
            cout<<endl;
        }


        //计算模糊矩阵U
        maxU=0;
        for(int i=0;i<k;i++){
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    double ss1=0,ss2=0;
                    ss1=1.0/(1-RBFfunction(featureMat[r][c],C[i],150));
                    for(int j=0;j<k;j++){
                        double sss=1.0/(1-RBFfunction(featureMat[r][c],C[j],150));
                        ss2+=sss;
                    }


                    double dd=abs(U[i][num]-ss1/ss2);
                    if(dd>maxU){
                        maxU=dd;
                    }

                    U[i][num]=ss1/ss2;

                    num++;
                }
            }
        }

        //        for(int i=0;i<k;i++){
        //            cout<<U[i][0]<<" ";
        //        }
        //        cout<<endl;


        cout<<"maxU:"<<maxU<<endl;
        cout<<"************************************************************"<<endl;

    }

    cout<<"end"<<endl;

    //    for(int j=0;j<size;j++){
    //        double ss=0;
    //        for(int i=0;i<k;i++){
    //            cout<<U[i][j]<<" ";
    //        }
    //        cout<<endl;

    //    }

    int step=255/k;
    Mat changeImg(img.rows,img.cols,CV_8UC3);
    int kkk=0;
    Vec3b* color=new Vec3b[k];
    for(int i=0;i<k;i++){
        color[i]={uchar(step*i),uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2))};
        cout<<color[i]<<" ";
    }


    img_type=Mat(img.rows,img.cols,CV_8U);
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int type=0;
            double maxK=0;
            for(int m=0;m<k;m++){
                if(U[m][kkk]>maxK){
                    maxK=U[m][kkk];
                    type=m;
                }
            }
            img_type.at<uchar>(i,j)=type;
            Vec3b ccc=color[type];
            changeImg.at<Vec3b>(i,j)=ccc;
            kkk++;
        }
    }
    return changeImg;

}

//注：聚类中心最好为浮点型
Mat FCM_Road_Kernel_Space(Mat img,int k,Mat directImg,Mat *meanSquareImg,Mat *gaborImg,Mat *gaborImg2,Mat *gaborImg3){
    int m=2,maxStep=20; //m默认为2
    double maxU=1000;
    int size=img.rows*img.cols;
    //    cout<<size<<endl;
    double **U=new double*[k];
    for(int i=0;i<k;i++){
        U[i]=new double[size];
    }
    double **H=new double*[k]; //空间信息
    for(int i=0;i<k;i++){
        H[i]=new double[size];
    }
    double **UU=new double*[k];//加入空间信息后的U
    for(int i=0;i<k;i++){
        UU[i]=new double[size];
    }


    double **C;
    double **CC;
    //构建图像特征矩阵（用于聚类）
    double ***featureMat;
    featureMat=new double**[img.rows];
    for(int i=0;i<img.rows;i++){
        featureMat[i]=new double*[img.cols];
        for(int j=0;j<img.cols;j++){
            featureMat[i][j]=new double[FeatureNum];
            int minDirect=directImg.at<uchar>(i,j);
            featureMat[i][j][0]=meanSquareImg[minDirect].at<float>(i,j);
            featureMat[i][j][1]=gaborImg[minDirect].at<float>(i,j);
            //            featureMat[i][j][2]=gaborImg2[minDirect].at<float>(i,j);
            //            featureMat[i][j][3]=gaborImg3[minDirect].at<float>(i,j);
            featureMat[i][j][2]=gaborImg[minDirect].at<float>(i,j);
            featureMat[i][j][3]=meanSquareImg[minDirect].at<float>(i,j);
        }
    }
    //特征归一化
    double sumFeature[4]={0,0,0,0};
    double minFeature[4]={1000,1000,1000,1000},maxFeature[4]={0,0,0,0};
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            for(int z=0;z<4;z++){
                if(minFeature[z]>featureMat[i][j][z]){
                    minFeature[z]=featureMat[i][j][z];
                }
                if(maxFeature[z]<featureMat[i][j][z]){
                    maxFeature[z]=featureMat[i][j][z];
                }
            }

        }
    }
    for(int i=0;i<4;i++){
        cout<<maxFeature[i]<<" ";
    }
    cout<<endl;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            for(int z=0;z<4;z++){
                featureMat[i][j][z]=(featureMat[i][j][z]-minFeature[z])/(maxFeature[z]-minFeature[z]);
                sumFeature[z]+=featureMat[i][j][z];
            }
        }
    }

    //初始化隶属度矩阵
    C=new double*[k];
    for(int i=0;i<k;i++){
        C[i]=new double[FeatureNum];
    }

    CC=new double*[k];
    for(int i=0;i<k;i++){
        CC[i]=new double[FeatureNum];
    }

    randCent_Road(C,k);

    for(int i=0;i<k;i++){
        for(int j=0;j<FeatureNum;j++){
            cout<<C[i][j]<<" ";
        }
        cout<<endl;
    }

    for(int i=0;i<k;i++){
        int num=0;
        for(int r=0;r<img.rows;r++){
            for(int c=0;c<img.cols;c++){
                double ss1=0,ss2=0;
                ss1=1.0/(1-RBFfunction(featureMat[r][c],C[i],150));
                for(int j=0;j<k;j++){
                    double sss=1.0/(1-RBFfunction(featureMat[r][c],C[j],150));
                    ss2+=sss;
                }

                U[i][num]=ss1/ss2;

                num++;
            }
        }
    }
    //    for(int j=0;j<size;j++){
    //        double ss=0;
    //        for(int i=0;i<k;i++){
    //            ss+=U[i][j];
    //        }
    //        cout<<ss<<endl;
    //    }



    cout<<"************************************************************"<<endl;

    int stepk=0;
    while(maxU>0.01&&stepk<20){
        stepk++;
        //计算聚类中心C
        for(int i=0;i<k;i++){
            double *sum1=new double[FeatureNum]();
            double *sum2=new double[FeatureNum]();
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    double uuu=U[i][num]*U[i][num];
                    for(int z=0;z<FeatureNum;z++){
                        sum1[z]+=uuu*featureMat[r][c][z]*RBFfunction(featureMat[r][c],C[i],150);
                        sum2[z]+=uuu*RBFfunction(featureMat[r][c],C[i],150);
                    }

                    num++;
                }
            }

            for(int z=0;z<FeatureNum;z++){
                C[i][z]=sum1[z]/sum2[z];
                cout<<C[i][z]<<" ";
            }
            cout<<endl;
        }


        //计算模糊矩阵U
        maxU=0;
        for(int i=0;i<k;i++){
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    double ss1=0,ss2=0;
                    ss1=1.0/(1-RBFfunction(featureMat[r][c],C[i],150));
                    for(int j=0;j<k;j++){
                        double sss=1.0/(1-RBFfunction(featureMat[r][c],C[j],150));
                        ss2+=sss;
                    }


                    double dd=abs(U[i][num]-ss1/ss2);
                    if(dd>maxU){
                        maxU=dd;
                    }

                    U[i][num]=ss1/ss2;

                    num++;
                }
            }
        }

        //        for(int i=0;i<k;i++){
        //            cout<<U[i][0]<<" ";
        //        }
        //        cout<<endl;


        cout<<"maxU:"<<maxU<<endl;
        cout<<"************************************************************"<<endl;

        //加入空间信息
        //求H(得先把U的行展开？？)
        double** imgU=new double*[img.rows];
        for(int i=0;i<img.rows;i++){
            imgU[i]=new double[img.cols];
        }
        for(int i=0;i<k;i++){
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    imgU[r][c]=U[i][num];
                    num++;
                }
            }

            num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    int x=r;
                    int y=c;
                    if(x<=0){
                        x=1;
                    }
                    if(x>=img.rows-1){
                        x=img.rows-2;
                    }
                    if(y<=0){
                        y=1;
                    }
                    if(y>=img.cols-1){
                        y=img.cols-2;
                    }
                    double zz=0;
                    for(int m=-1;m<2;m++){
                        for(int n=-1;n<2;n++){
                            zz+=imgU[x+m][y+n];
                        }
                    }
                    zz/=9.0;
                    H[i][num]=zz;

                    num++;
                }
            }
        }

        //求UU
        for(int i=0;i<k;i++){
            for(int j=0;j<size;j++){

                double sss1=U[i][j]*H[i][j],sss2=0;
                for(int z=0;z<k;z++){
                    sss2+=U[z][j]*H[z][j];
                }
                UU[i][j]=sss1/sss2;
            }
        }

        //求CC
        for(int i=0;i<k;i++){
            double *sum1=new double[FeatureNum]();
            double *sum2=new double[FeatureNum]();
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    double uuu=UU[i][num]*UU[i][num];
                    for(int z=0;z<FeatureNum;z++){
                        sum1[z]+=uuu*featureMat[r][c][z]*RBFfunction(featureMat[r][c],C[i],150);
                        sum2[z]+=uuu*RBFfunction(featureMat[r][c],C[i],150);
                    }

                    num++;
                }
            }

//            for(int z=0;z<FeatureNum;z++){
//                C[i][z]=sum1[z]/sum2[z];
//                cout<<C[i][z]<<" ";
//            }
//            cout<<endl;
        }

        //U与C的复制
        for(int i=0;i<k;i++){
            for(int j=0;j<size;j++){
                U[i][j]=UU[i][j];
            }
        }
        for(int i=0;i<k;i++){
            for(int j=0;j<FeatureNum;j++){
                C[i][j]=CC[i][j];
            }
        }

    }

    cout<<"end"<<endl;

    //    for(int j=0;j<size;j++){
    //        double ss=0;
    //        for(int i=0;i<k;i++){
    //            cout<<U[i][j]<<" ";
    //        }
    //        cout<<endl;

    //    }

    int step=255/k;
    Mat changeImg(img.rows,img.cols,CV_8UC3);
    int kkk=0;
    Vec3b* color=new Vec3b[k];
    for(int i=0;i<k;i++){
        color[i]={uchar(step*i),uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2))};
        cout<<color[i]<<" ";
    }


    img_type=Mat(img.rows,img.cols,CV_8U);
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int type=0;
            double maxK=0;
            for(int m=0;m<k;m++){
                if(UU[m][kkk]>maxK){
                    maxK=UU[m][kkk];
                    type=m;
                }
            }
            img_type.at<uchar>(i,j)=type;
            Vec3b ccc=color[type];
            changeImg.at<Vec3b>(i,j)=ccc;
            kkk++;
        }
    }
    return changeImg;

}
