void MainWindow::DTreeTnterpolationShow(int row,int col){
    cv::Mat img=cv::imread(fileName.toStdString());

    float Row_B = row;
    float Col_B = col;

    Mat changeImg(Row_B,Col_B,CV_8UC3);
    for(int i=0;i<Row_B;i++){
        for(int j=0;j<Col_B;j++){
            float x=i*(img.rows/Row_B);
            float y=j*(img.cols/Col_B);
            if(x<1){
                x=1;
            }
            if(y<1){
                y=1;
            }
            if(x>img.rows-3){
                x=img.rows-3;
            }
            if(y>img.cols-3){
                y=img.cols-3;
            }


            float w_x[4], w_y[4];//行列方向的加权系数
            getW_x(w_x, x);
            getW_y(w_y, y);

            Vec3f temp={0,0,0};
            for(int s=0;s<=3;s++){
                for(int t=0;t<=3;t++){
                    temp = temp + (Vec3f)(img.at<Vec3b>(int(x) + s - 1, int(y) + t - 1))*w_x[s] * w_y[t];
                }
            }

            changeImg.at<Vec3b>(i, j) = (Vec3b)temp;

        }
    }
    changeImage=changeImg;
    Mat image_RBG;
    cv::cvtColor(changeImage,image_RBG,CV_BGR2RGB);
    QImage imgs = QImage((const unsigned char*)(image_RBG.data),image_RBG.cols,image_RBG.rows,image_RBG.cols*image_RBG.channels(),QImage::Format_RGB888);
    ui->showImg->clear();
    ui->showImg->setPixmap(QPixmap::fromImage(imgs));
    //        ui->showImg->resize(ui->showImg->pixmap()->size());
    ui->showImg->show();

    waitKey(0);
}