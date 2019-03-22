//#include <opencv2/opencv.hpp>
//#include<iostream>
//#include<algorithm>

//using namespace std;
//using namespace cv;

//Mat soble(Mat img);
//double getSim(Vec3b a,Vec3b b);
//Vec3b* randCent(Mat img,int k);
//Mat KMeans(Mat img,int k);


//int main(){
//    Mat img=imread("h:\\hhh2.png");
//    int k=3;
////    Mat img2=soble(img);
//    Mat img2=KMeans(img,k);
//    imshow("img",img);
//    imshow("img2",img2);
//    waitKey();
//    return 0;
//}

//Mat soble(Mat img){
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

//            int filterX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
//            int filterY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};

//            double fx=0.0,fy=0.0;
//            for(int m=-1;m<2;m++){
//                for(int n=-1;n<2;n++){
//                    fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[0];
//                    fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[0];
//                    fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[1];
//                    fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[1];
//                    fx+=filterX[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[2];
//                    fy+=filterY[m+1][n+1]*img.at<Vec3b>(x+m,y+n)[2];
//                }
//            }
//            fx=fx/3.0;
//            fy=fy/3.0;
//            double f=fabs(fx)+fabs(fy);

//            Vec3b vecf={(uchar)f,(uchar)f,(uchar)f};
//            changeImg.at<Vec3b>(i,j)=vecf;

//        }
//    }
//    return changeImg;

//}

//double getSim(Vec3b a,Vec3b b){
//    double sum=0;
//    for(int i=0;i<3;i++){
//        sum+=(a[i]-b[i])*(a[i]-b[i]);
//    }
//    return sqrt(sum);
//}

//Vec3b* randCent(Mat img,int k){
//    srand((unsigned)time(NULL));
//    Vec3b* centroids=new Vec3b(k);
//    for(int i=0;i<3;i++){
//        int min=255,max=0;
//        for(int r=0;r<img.rows;r++){
//            for(int c=0;c<img.cols;c++){
//                uchar cc=img.at<Vec3b>(r,c)[i];
//                if(cc<min){
//                    min=cc;
//                }
//                if(cc>max){
//                    max=cc;
//                }
//            }
//        }

//        for(int j=0;j<k;j++){
//            centroids[j][i]=rand()%(max-min)+min;
//        }
//    }
//    return centroids;
//}


//Mat KMeans(Mat img,int k){
//    int size=img.rows*img.cols;
//    int*  clusterLabel=new int[size];
//    Vec3b* centroids=randCent(img,k);
//    bool change=true;
//    int counter=0;



//    while(counter<=50){
//        counter++;
//        cout<<"1"<<endl;
//        change=false;
//        int kkk=0;
//        for(int r=0;r<img.rows;r++){
//            for(int c=0;c<img.cols;c++){
//                double minDis=10000000;
//                int minIndex=-1;
//                Vec3b ccc=img.at<Vec3b>(r,c);
//                for(int j=0;j<k;j++){
//                    double distJI=getSim(centroids[j],ccc);
//                    if(distJI<minDis){
//                        minDis=distJI;
//                        minIndex=j;
//                    }
//                    if(clusterLabel[kkk]!=minIndex){
//                        change=true;
//                    }
//                    clusterLabel[kkk]=minIndex;
//                }

//                kkk++;
//            }
//        }
//        cout<<2<<endl;
//        if(change){

//            for(int i=0;i<k;i++){
//                kkk=0;
//                int num=1;
//                int sum1=0,sum2=0,sum3=0;
//                for(int r=0;r<img.rows;r++){
//                    for(int c=0;c<img.cols;c++){
//                        if(clusterLabel[kkk]==i){
//                            Vec3b ccc=img.at<Vec3b>(r,c);
//                            sum1+=ccc[0];
//                            sum2+=ccc[1];
//                            sum3+=ccc[2];
//                            num++;
//                        }

//                        kkk++;
//                    }
//                }
//                sum1=sum1/num;
//                sum2=sum2/num;
//                sum3=sum3/num;
//                centroids[i]={sum1,sum2,sum3};
//                cout<<centroids[i]<<endl;
//            }
//        }else{
//            cout<<"GGGG"<<endl;
//            break;
//        }
//    }

//    int step=255/k;
//    Mat changeImg(img.rows,img.cols,CV_8UC3);
//    int kkk=0;
//    cout<<3<<endl;
//    Vec3b* color=new Vec3b[k];
//    for(int i=0;i<k;i++){
//        color[i]={uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2)),uchar(step*i*(rand()%2))};
//    }
//    for(int i=0;i<img.rows;i++){
//        for(int j=0;j<img.cols;j++){
//            int type=int(clusterLabel[kkk]);
//            Vec3b ccc=color[type];
//            changeImg.at<Vec3b>(i,j)=ccc;
//            kkk++;
//        }
//    }

//    return changeImg;
//}









