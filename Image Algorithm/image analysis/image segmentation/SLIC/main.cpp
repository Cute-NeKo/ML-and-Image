/*
 * test_slic.cpp.
 *
 * Written by: Pascal Mettes.
 *
 * This file creates an over-segmentation of a provided image based on the SLIC
 * superpixel algorithm, as implemented in slic.h and slic.cpp.
 */
#include<opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;

#include "slic.h"

int main() {
    cv::Mat img=cv::imread("H:\\95E5B1206DFFF4A44A195DCBEC0DA446.jpg");
    /* Load the image and convert to Lab colour space. */
    IplImage imgTmp = img;
    IplImage *image = cvCloneImage(&imgTmp);
    IplImage *lab_image = cvCloneImage(image);
    cvCvtColor(image, lab_image, CV_BGR2Lab);

    /* Yield the number of superpixels and weight-factors from the user. */
    int w = image->width, h = image->height;
    int nr_superpixels = 200;
    int nc = 40;

    double step = sqrt((w * h) / (double) nr_superpixels);

    /* Perform the SLIC superpixel algorithm. */
    Slic slic;

    slic.generate_superpixels(lab_image, step, nc);
    slic.create_connectivity(lab_image);

    /* Display the contours and show the result. */
    slic.display_contours(image, CV_RGB(255,0,0));
//    slic.display_center_grid(image, CV_RGB(255,0,0));
//    cvShowImage("result", image);

    cv::Mat resultImg=cv::cvarrToMat(image);
    cout<<resultImg.type()<<endl;
    cv::imshow("Gg",resultImg);
    cvWaitKey(0);
}
