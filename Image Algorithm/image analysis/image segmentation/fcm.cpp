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

int main(){
   Mat img =imread("h://my.jpg");
   //    img=meanFilter(img);
//    medianBlur(img,img,5);
   Mat imgt;
   img.convertTo(imgt,CV_64FC3,1);
   int k=2;

   Mat img2=FCMS(imgt,k);

   imshow("img1",img);
   imshow("img2",img2);

   waitKey();
   return 0;

}


double getSim_d(Vec3d a,Vec3d b){
   double sum=0;
   for(int i=0;i<3;i++){
       sum+=abs((a[i]-b[i])*(a[i]-b[i]));
   }
   return sqrt(sum);
}

Vec3d* randCent2(Mat img,int k){
   srand((unsigned)time(NULL));
   Vec3d* centroids=new Vec3d(k);
   for(int i=0;i<3;i++){
       int min=255,max=0;
       for(int r=0;r<img.rows;r++){
           for(int c=0;c<img.cols;c++){
               uchar cc=img.at<Vec3d>(r,c)[i];
               if(cc<min){
                   min=cc;
               }
               if(cc>max){
                   max=cc;
               }

           }
       }


       for(int j=0;j<k;j++){
           centroids[j][i]=rand()%(max-min)+min;
       }
   }
   return centroids;
}


//注：聚类中心最好为浮点型
Mat FCM(Mat img,int k){
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

   while(maxU>0.01){
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
       color[i]={uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2))};
       cout<<color[i]<<" ";
   }

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
           Vec3b ccc=color[type];
           changeImg.at<Vec3b>(i,j)=ccc;
           kkk++;
       }
   }
   return changeImg;

}

//fcm改进算法
Mat FCMS(Mat img,int k){
   //对图像进行过滤波
   Mat meanImg(img.rows,img.cols,CV_64FC3);

   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int x=i;
           int y=j;
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

           double filter[3][3]={{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}};

           double b=0,g=0,r=0;
           for(int m=-1;m<2;m++){
               for(int n=-1;n<2;n++){
                   b+=img.at<Vec3d>(x+m,y+n)[0]*filter[m+1][n+1];
                   g+=img.at<Vec3d>(x+m,y+n)[1]*filter[m+1][n+1];
                   r+=img.at<Vec3d>(x+m,y+n)[2]*filter[m+1][n+1];
               }
           }


           Vec3b vecf={b,g,r};
           meanImg.at<Vec3d>(i,j)=vecf;

       }
   }
   /*********************************************************************/

   int m=2,maxStep=10; //m默认为2，默认迭代次数为10（其实用不到）
   double a=0.5;   //控制系数
   double maxU=1000;
   int size=img.rows*img.cols;
   //    cout<<size<<endl;
   double **U=new double*[k];
   for(int i=0;i<k;i++){
       U[i]=new double[size];
   }
   Vec3d *C=new Vec3d[k];


   //初始化隶属度矩阵
   C=randCent2(img,k);

   for(int i=0;i<k;i++){
       cout<<C[i]<<endl;
   }

   for(int i=0;i<k;i++){
       int num=0;
       for(int r=0;r<img.rows;r++){
           for(int c=0;c<img.cols;c++){
               Vec3d pix=img.at<Vec3d>(r,c);
               Vec3d pixmean=meanImg.at<Vec3d>(r,c);
               double ss1=0,ss2=0,ss=0;
               ss1=getSim_d(pix,C[i])*getSim_d(pix,C[i])+a*getSim_d(pixmean,C[i])*getSim_d(pixmean,C[i]);
               //                for(int j=0;j<k;j++){
               //                    ss2+=(getSim_d(pix,C[j])*getSim_d(pix,C[j])+a*getSim_d(pixmean,C[j])*getSim_d(pixmean,C[j]));
               //                }


               //                U[i][num]=ss1/ss2;

               //                cout<<ss1<<" "<<ss2<<endl;
               for(int j=0;j<k;j++){

                   ss2=(getSim_d(pix,C[j])*getSim_d(pix,C[j])+a*getSim_d(pixmean,C[j])*getSim_d(pixmean,C[j]));
                   ss+=(ss1/ss2);
               }
               U[i][num]=1.0/ss;

               num++;
           }
       }
   }
   //    for(int j=0;j<size;j++){
   //        double ss=0;
   //        for(int i=0;i<k;i++){
   //            //                       ss+=U[i][j];
   //            cout<<U[i][j]<<" ";
   //        }
   //        //                   cout<<ss<<" ";
   //        cout<<endl;
   //    }
   //    cout<<endl;


   cout<<"************************************************************"<<endl;

   int stepAll=0;
   while(maxU>0.01){
       //计算聚类中心C
       for(int i=0;i<k;i++){
           Vec3d sum1={0,0,0};
           Vec3d sum2={0,0,0};
           int num=0;
           for(int r=0;r<img.rows;r++){
               for(int c=0;c<img.cols;c++){
                   double sss=U[i][num]*U[i][num];
                   Vec3d pix=img.at<Vec3d>(r,c);
                   Vec3d pixmean=meanImg.at<Vec3d>(r,c);

                   sum1+=(sss*(pix+a*pixmean));

                   Vec3d sss2={sss,sss,sss};
                   sum2+=sss2;

                   num++;
               }
           }
           sum2=(1+a)*sum2;
           //            cout<<sum1<<" "<<sum2<<endl;

           C[i]={sum1[0]/sum2[0],sum1[1]/sum2[1],sum1[2]/sum2[2]};
           cout<<C[i]<<endl;
       }


       //计算模糊矩阵U
       maxU=0;
       for(int i=0;i<k;i++){
           int num=0;
           for(int r=0;r<img.rows;r++){
               for(int c=0;c<img.cols;c++){
                   Vec3d pix=img.at<Vec3d>(r,c);
                   Vec3d pixmean=meanImg.at<Vec3d>(r,c);
                   double ss1=0,ss2=0,ss=0;

                   ss1=getSim_d(pix,C[i])*getSim_d(pix,C[i])+a*getSim_d(pixmean,C[i])*getSim_d(pixmean,C[i]);
                   for(int j=0;j<k;j++){

                       ss2=(getSim_d(pix,C[j])*getSim_d(pix,C[j])+a*getSim_d(pixmean,C[j])*getSim_d(pixmean,C[j]));
                       ss+=(ss1/ss2);
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

       //        for(int j=0;j<1;j++){
       //            double ss=0;
       //            for(int i=0;i<k;i++){
       //                cout<<U[i][j]<<" ";
       //            }
       //            cout<<endl;

       //        }

       stepAll++;
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
   Mat changeImg2(img.rows,img.cols,CV_8UC3);
   int kkk=0;
   Vec3b* color=new Vec3b[k];
   for(int i=0;i<k;i++){
       color[i]={uchar(step*i),uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2))};
       cout<<color[i]<<" ";
   }

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
           Vec3b ccc=color[type];
           changeImg2.at<Vec3b>(i,j)=ccc;

           kkk++;
       }
   }
   return changeImg2;


}

//基于像素相关性改进的FCM
Mat FCMSPixel(Mat img,int k){
   int size=img.rows*img.cols;

   //计算相关性矩阵
   double LambdaS=3,LambdaG=3;
   double **R;
   R=new double*[size];
   for(int i=0;i<size;i++){
       R[i]=new double[25];
   }

   int num=0;
   for(int i=0;i<img.rows;i++){
       for(int j=0;j<img.cols;j++){
           int x=i;
           int y=j;
           if(x<=1){
               x=2;
           }
           if(x>=img.rows-2){
               x=img.rows-3;
           }
           if(y<=1){
               y=2;
           }
           if(y>=img.cols-2){
               y=img.cols-3;
           }

           double sigmaG=0;
           //求平法灰度方差
           for(int m=-2;m<3;m++){
               for(int n=-2;n<3;n++){
                   sigmaG+=pow(differRGB(img.at<Vec3d>(x,y),img.at<Vec3d>(x+m,y+n)),2);
               }
           }
           sigmaG/=25.0;
           if(sigmaG==0){
               sigmaG=0.0001;
           }


           double RS[25],RG[25];
           int num2=0;
           for(int m=-2;m<3;m++){
               for(int n=-2;n<3;n++){
                   double ss=sqrt(m*m+n*n);
                   RS[num2]=exp(-ss/LambdaS);
                   if(m==0&&n==0){
                       RS[num2]=0;
                   }

                   RG[num2]=exp(-pow(differRGB(img.at<Vec3d>(x,y),img.at<Vec3d>(x+m,y+n)),2)/(LambdaG*sigmaG));

                   num2++;
               }
           }
           for(int m=0;m<25;m++){
               R[num][m]=RS[m]*RG[m];

           }



           num++;
       }
   }
   //    for(int i=0;i<size;i++){
   //        for(int j=0;j<25;j++){
   //            cout<<R[i][j]<<" ";
   //            if((j+1)%5==0){
   //                cout<<endl;
   //            }
   //        }
   //        cout<<endl;
   //    }

   cout<<"start"<<endl;
   //FCM算法开始
   /*********************************************************************/

   int m=2,maxStep=10; //m默认为2，默认迭代次数为10（其实用不到）
   double a=0.5;   //控制系数
   double maxU=1000;
   double **U=new double*[k];
   for(int i=0;i<k;i++){
       U[i]=new double[size];
   }
   Vec3d *C=new Vec3d[k];


   //初始化隶属度矩阵
   C=randCent2(img,k);

   for(int i=0;i<k;i++){
       cout<<C[i]<<endl;
   }

   for(int i=0;i<k;i++){
       int num=0;
       for(int r=0;r<img.rows;r++){
           for(int c=0;c<img.cols;c++){

               int x=r;
               int y=c;
               if(x<=1){
                   x=2;
               }
               if(x>=img.rows-2){
                   x=img.rows-3;
               }
               if(y<=1){
                   y=2;
               }
               if(y>=img.cols-2){
                   y=img.cols-3;
               }
               ////////////

               double ss1=0,ss2=0,ss=0;
               double tt=0;
               int num2=0;

               for(int m=-2;m<3;m++){
                   for(int n=-2;n<3;n++){
                       tt+=(R[num][num2]*pow(getSim_d(C[i],img.at<Vec3d>(x+m,y+n)),2));

                       num2++;
                   }
               }
               ///////////////////////////////
               ss1=(pow(getSim_d(C[i],img.at<Vec3d>(r,c)),2)+tt);
               for(int j=0;j<k;j++){

                   double tt2=0;num2=0;
                   for(int m=-2;m<3;m++){
                       for(int n=-2;n<3;n++){
                           tt2+=(R[num][num2]*pow(getSim_d(C[j],img.at<Vec3d>(x+m,y+n)),2));

                           num2++;
                       }
                   }
                   ss2=(pow(getSim_d(C[j],img.at<Vec3d>(r,c)),2)+tt2);
                   ss+=(ss1/ss2);

               }

               U[i][num]=1.0/ss;

               num++;
           }
       }
   }
//    for(int j=0;j<size;j++){
//        double ss=0;
//        for(int i=0;i<k;i++){
//            //            ss+=U[i][j];
//            cout<<U[i][j]<<" ";
//        }
//        //        cout<<ss<<" ";
//        cout<<endl;
//    }
//    cout<<endl;


   cout<<"************************************************************"<<endl;

       int stepAll=0;
       while(maxU>0.01){
           //计算聚类中心C
           for(int i=0;i<k;i++){
               Vec3d sum1={0,0,0};
               Vec3d sum2={0,0,0};
               int num=0;
               for(int r=0;r<img.rows;r++){
                   for(int c=0;c<img.cols;c++){
                       int x=r;
                       int y=c;
                       if(x<=1){
                           x=2;
                       }
                       if(x>=img.rows-2){
                           x=img.rows-3;
                       }
                       if(y<=1){
                           y=2;
                       }
                       if(y>=img.cols-2){
                           y=img.cols-3;
                       }
                       ////////////

                       double sss=U[i][num]*U[i][num];
                       Vec3d tt1={0,0,0};
                       double tt2=0;
                       int num2=0;

                       for(int m=-2;m<3;m++){
                           for(int n=-2;n<3;n++){
                               tt1+=R[num][num2]*img.at<Vec3d>(x+m,y+n);
                               tt2+=R[num][num2];

                               num2++;
                           }
                       }

                       Vec3d pix=img.at<Vec3d>(r,c);

                       sum1+=(sss*(pix+tt1));

                       Vec3d sss2={sss*(1+tt2),sss*(1+tt2),sss*(1+tt2)};
                       sum2+=sss2;

                       num++;
                   }
               }

               //            cout<<sum1<<" "<<sum2<<endl;

               C[i]={sum1[0]/sum2[0],sum1[1]/sum2[1],sum1[2]/sum2[2]};
               cout<<C[i]<<endl;
           }


           //计算模糊矩阵U
           maxU=0;
           for(int i=0;i<k;i++){
               int num=0;
               for(int r=0;r<img.rows;r++){
                   for(int c=0;c<img.cols;c++){

                       int x=r;
                       int y=c;
                       if(x<=1){
                           x=2;
                       }
                       if(x>=img.rows-2){
                           x=img.rows-3;
                       }
                       if(y<=1){
                           y=2;
                       }
                       if(y>=img.cols-2){
                           y=img.cols-3;
                       }
                       ////////////

                       double ss1=0,ss2=0,ss=0;
                       double tt=0;
                       int num2=0;

                       for(int m=-2;m<3;m++){
                           for(int n=-2;n<3;n++){
                               tt+=(R[num][num2]*pow(getSim_d(C[i],img.at<Vec3d>(x+m,y+n)),2));

                               num2++;
                           }
                       }
                       ///////////////////////////////
                       ss1=(pow(getSim_d(C[i],img.at<Vec3d>(r,c)),2)+tt);
                       for(int j=0;j<k;j++){

                           double tt2=0;num2=0;
                           for(int m=-2;m<3;m++){
                               for(int n=-2;n<3;n++){
                                   tt2+=(R[num][num2]*pow(getSim_d(C[j],img.at<Vec3d>(x+m,y+n)),2));

                                   num2++;
                               }
                           }
                           ss2=(pow(getSim_d(C[j],img.at<Vec3d>(r,c)),2)+tt2);
                           ss+=(ss1/ss2);

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

           //        for(int j=0;j<1;j++){
           //            double ss=0;
           //            for(int i=0;i<k;i++){
           //                cout<<U[i][j]<<" ";
           //            }
           //            cout<<endl;

           //        }

           stepAll++;
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
       Mat changeImg2(img.rows,img.cols,CV_8UC3);
       int kkk=0;
       Vec3b* color=new Vec3b[k];
       for(int i=0;i<k;i++){
           color[i]={uchar(step*i),uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2))};
           cout<<color[i]<<" ";
       }

       //保存分类情况用于后分割
       int** typeImg=new int*[img.rows];
       for(int i=0;i<img.rows;i++){
           typeImg[i]=new int[img.cols];
       }



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
               Vec3b ccc=color[type];
               changeImg2.at<Vec3b>(i,j)=ccc;
               typeImg[i][j]=type;

               kkk++;
           }
       }

       //对像素进行重分类
//        for(int i=0;i<img.rows;i++){
//            for(int j=0;j<img.cols;j++){
//                int x=i;
//                int y=j;
//                if(x<=1){
//                    x=2;
//                }
//                if(x>=img.rows-2){
//                    x=img.rows-3;
//                }
//                if(y<=1){
//                    y=2;
//                }
//                if(y>=img.cols-2){
//                    y=img.cols-3;
//                }
//                ////////////
//                int Nj=0;
//                for(int m=-2;m<3;m++){
//                    for(int n=-2;n<3;n++){
//                        if(typeImg[x+m][y+n]==typeImg[x][y]){
//                            Nj++;
//                        }
//                    }
//                }
//                if(Nj){

//                }



//            }
//        }




       return changeImg2;


}

double differRGB(Vec3d c1,Vec3d c2){
   double sum=0;
   sum+=abs(c1[0]-c2[0]);
   sum+=abs(c1[1]-c2[1]);
   sum+=abs(c1[2]-c2[2]);
   return sum/3.0;
}









