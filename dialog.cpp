#include "dialog.h"
#include "ui_dialog.h"
#include "opencv2/opencv.hpp"

#include <QFileDialog>
#include <QDebug>

Dialog::Dialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::Dialog)
{
    ui->setupUi(this);
}

Dialog::~Dialog()
{
    delete ui;
}


void Dialog::on_pushButton_open_clicked()
{
    readImg();

    meanThreshold(grayImg);
}

void Dialog::readImg()
{
    QString imgName = QFileDialog::getOpenFileName(this, "Open a file", QDir::currentPath().append("/images"));

    qDebug() << imgName;
    origImg = cv::imread(imgName.toStdString());
    grayImg = cv::imread(imgName.toStdString(), cv::IMREAD_GRAYSCALE);

    QImage q_origImg(origImg.data, origImg.cols, origImg.rows, origImg.step, QImage::Format_BGR888);
    QImage q_grayImg(grayImg.data, grayImg.cols, grayImg.rows, grayImg.step, QImage::Format_Grayscale8);

    q_origImg = q_origImg.scaled(700, 400, Qt::KeepAspectRatio);
    q_grayImg = q_grayImg.scaled(700, 400, Qt::KeepAspectRatio);

    // display orig image
    ui->label_orig->setPixmap(QPixmap::fromImage(q_origImg));
    // display gray image
    ui->label_gray->setPixmap(QPixmap::fromImage(q_grayImg));
}

double Dialog::meanThreshold(cv::Mat src_Mat)
{
    // calculate Mean
    cv::Scalar tmp_mean;
    tmp_mean = cv::mean(src_Mat);
    //qDebug() << tmp_mean.val[0];

    double threshold = tmp_mean.val[0];
    double thresholdNext = 0;
    bool done = (qAbs(threshold - thresholdNext) < 0.255);

    cv::Mat dst_Mat1(src_Mat.rows, src_Mat.cols, src_Mat.type());
    cv::Mat dst_Mat2(src_Mat.rows, src_Mat.cols, src_Mat.type());

    while(!done)
    {
        cv::threshold(src_Mat, dst_Mat1, threshold, 255, cv::THRESH_TOZERO);
        cv::threshold(src_Mat, dst_Mat2, threshold, 255, cv::THRESH_TOZERO_INV);

        cv::Scalar tmp_mean1, tmp_stddev1, tmp_mean2, tmp_stddev2;
        //cv::meanStdDev(dst_Mat1, tmp_mean1, tmp_stddev1);
        //cv::meanStdDev(dst_Mat2, tmp_mean2, tmp_stddev2);
        // Tnext = (1/(s1+s2))*(s1*mean(fd(gd)) + s2*mean(fd(~gd)));
        // Tnext = (0.5)*(mean(fd(gd)) + mean(fd(~gd)));

        //qDebug() << calculateMean(dst_Mat1) /255;
        //thresholdNext = (tmp_mean1.val[0] + tmp_mean2.val[0]) /2;
        thresholdNext = ((calculateMean(dst_Mat1) + calculateMean(dst_Mat2)));
        thresholdNext = thresholdNext /2;
        qDebug() << ((calculateMean(dst_Mat1) + calculateMean(dst_Mat2)) /2)/255;
        qDebug() << threshold /255 << calculateMean(dst_Mat1) /255 << calculateMean(dst_Mat2) /255 << thresholdNext /255;
        //qDebug() << threshold << calculateMean(dst_Mat1) << calculateMean(dst_Mat2) << thresholdNext;

        done = (qAbs(threshold - thresholdNext) < 0.255);
        threshold = thresholdNext;
    }
    qDebug() << threshold / 255;

    // dsplay
    QImage q_grayImg(dst_Mat1.data, dst_Mat1.cols, dst_Mat1.rows, dst_Mat1.step, QImage::Format_Grayscale8);
    QImage q_grayImg1(dst_Mat2.data, dst_Mat2.cols, dst_Mat2.rows, dst_Mat2.step, QImage::Format_Grayscale8);
    q_grayImg = q_grayImg.scaled(700, 400, Qt::KeepAspectRatio);
    q_grayImg1 = q_grayImg1.scaled(700, 400, Qt::KeepAspectRatio);
    ui->label_mean->setPixmap(QPixmap::fromImage(q_grayImg));
    ui->label_meanstd->setPixmap(QPixmap::fromImage(q_grayImg1));


    return threshold;
    /*
    while(!done)
    {
        qDebug() << src_Mat.at<uchar>(0, 0);
        //std::cout << src_Mat.at<cv::Vec3b>(0, 0)<< std::endl;

        int nChannels = src_Mat.channels();
        int nRows = src_Mat.rows;
        int nCols = src_Mat.cols * nChannels;
        int nStep = src_Mat.step;

        uchar* srcData= src_Mat.data;
        uchar* dstData1 = dst_Mat1.data;
        uchar* dstData2 = dst_Mat2.data;

        for( int j = 0; j < nRows; j++ ){
            for( int i = 0; i < nCols; i++ ) {
                if(*src_Mat.ptr(j, i) >= threshold)
                {
                    *(dstData1+i) = *(srcData+i);
                    *(dstData2+i) = 0;
                }
                else
                {
                    *(dstData1+i) = 0;
                    *(dstData2+i) = *(srcData+i);
                }
            }
            srcData += nStep;
            dstData1 += nStep;
            dstData2 += nStep;
        }
        //qDebug() << (QString::number(dst_Mat1.at<uchar>(0, 0)) > 50);
        qDebug() << *src_Mat.ptr(0, 0);
        qDebug() << dst_Mat1.at<uchar>(0, 0);
        qDebug() << dst_Mat2.at<uchar>(0, 0);

    }
    */

}

double Dialog::calculateMean(cv::Mat src_Mat)
{
    cv::Mat dst_Mat(src_Mat.rows, src_Mat.cols, src_Mat.type());

    int nChannels = src_Mat.channels();
    int nRows = src_Mat.rows;
    int nCols = src_Mat.cols * nChannels;
    int nStep = src_Mat.step;

    double sum;
    int count = 0;
    uchar* srcData= src_Mat.data;
    uchar* dstData = dst_Mat.data;
    for( int j = 0; j < nRows; j++ ){
        for( int i = 0; i < nCols; i++ ) {
            if(*src_Mat.ptr(j, i) != 0)
            {
                sum += *src_Mat.ptr(j, i);
                count++;
            }
        }
        srcData += nStep;
        dstData += nStep;
    }

    return sum / count;
}

void Dialog::on_pushButton_clicked()
{
    cv::Mat dst_Mat(grayImg.rows, grayImg.cols, grayImg.type());

    cv::threshold(grayImg, dst_Mat, meanThreshold(grayImg), 255, cv::THRESH_BINARY);

    QImage q_grayImg(dst_Mat.data, dst_Mat.cols, dst_Mat.rows, dst_Mat.step, QImage::Format_Grayscale8);

    q_grayImg = q_grayImg.scaled(700, 400, Qt::KeepAspectRatio);

    // display orig image
    ui->label_orig->setPixmap(QPixmap::fromImage(q_grayImg));

}
