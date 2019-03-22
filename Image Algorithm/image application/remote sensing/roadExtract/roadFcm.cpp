#include <opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

double getSim_d(Vec3d a,Vec3d b);
Mat FCM(Mat img,int k);
Vec3d* randCent2(Mat img,int k);
Mat FCMS(Mat img,int k);
Mat meanFilter(Mat img);
Mat FCMSPixel(Mat img,int k);
double differRGB(Vec3d c1,Vec3d c2);

Mat FCM_Road(Mat img,int k);

Mat img_type;
int main(){
    Mat img =imread("h://city2.png");
    img=meanFilter(img);

    Mat imgt;
    img.convertTo(imgt,CV_64FC3,1);
    int k=7;
    Mat img2=FCM_Road(imgt,k);

    imshow("img1",img);
    imshow("img2",img2);
    for(int i=0;i<k;i++){
        Mat img3(img.rows,img.cols,CV_8UC3);
        for(int m=0;m<img.rows;m++){
            for(int n=0;n<img.cols;n++){
                if(img_type.at<uchar>(m,n)==i){
                    Vec3b vvv={255,255,255};
                    img3.at<Vec3b>(m,n)=vvv;
                }
            }
        }
        imshow("img"+to_string(i+4),img3);
    }

    waitKey();
    return 0;

}


//注：聚类中心最好为浮点型
Mat FCM_Road(Mat img,int k){
    int m=2,maxStep=10; //m默认为2
    double maxU=1000;
    int size=img.rows*img.cols;
    //    cout<<size<<endl;
    double **U=new double*[k];
    for(int i=0;i<k;i++){
        U[i]=new double[size];
    }
    Vec3d *C=new Vec3d[k];
    double J=0;
    //初始化隶属度矩阵
    C=randCent2(img,k);

    for(int i=0;i<k;i++){
        cout<<C[i]<<endl;
    }

    for(int i=0;i<k;i++){
        int num=0;
        for(int r=0;r<img.rows;r++){
            for(int c=0;c<img.cols;c++){
                double ss=0;
                for(int j=0;j<k;j++){
                    double ss2=getSim_d(img.at<Vec3d>(r,c),C[i])/getSim_d(img.at<Vec3d>(r,c),C[j]);
                    ss+=ss2*ss2;

                }

                U[i][num]=1.0/ss;

                num++;
            }
        }
    }
    //    for(int j=0;j<size;j++){
    //        double ss=0;
    //        for(int i=0;i<k;i++){
    //            cout<<U[i][j]<<" ";
    //        }
    //    }


    cout<<"************************************************************"<<endl;

    int kkkk=0;
    while(maxU>0.01&&kkkk<20){
        kkkk++;
        //计算聚类中心C
        for(int i=0;i<k;i++){
            //            double sum11=0,sum12=0,sum13=0;
            //            double sum21=0,sum22=0,sum23=0;
            Vec3d sum1={0,0,0};
            Vec3d sum2={0,0,0};
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    sum1+=img.at<Vec3d>(r,c)*U[i][num]*U[i][num];
                    double sss=U[i][num]*U[i][num];
                    Vec3d sss2={sss,sss,sss};
                    sum2+=sss2;

                    num++;
                }
            }

            C[i]={sum1[0]/sum2[0],sum1[1]/sum2[1],sum1[2]/sum2[2]};
            cout<<C[i]<<endl;
        }

        //    //计算目标函数J
        //    J=0;
        //    for(int i=0;i<k;i++){
        //        int num=0;
        //        for(int r=0;r<img.rows;r++){
        //            for(int c=0;c<img.cols;c++){
        //                double ss1=U[i][num],ss2=getSim(img.at<Vec3d>(r,c),C[i]);
        //                J+=ss1*ss1*ss2*ss2;
        //                num++;
        //            }
        //        }
        //    }

        //计算模糊矩阵U
        maxU=0;
        for(int i=0;i<k;i++){
            int num=0;
            for(int r=0;r<img.rows;r++){
                for(int c=0;c<img.cols;c++){
                    double ss=0;
                    for(int j=0;j<k;j++){
                        double ss2=getSim_d(img.at<Vec3d>(r,c),C[i])/getSim_d(img.at<Vec3d>(r,c),C[j]);

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

        //        for(int j=0;j<size;j++){
        //            double ss=0;
        //            for(int i=0;i<k;i++){
        //                cout<<U[i][j]<<" ";
        //            }
        //            cout<<endl;

        //        }

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
