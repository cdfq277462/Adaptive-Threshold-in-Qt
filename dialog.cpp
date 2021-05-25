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
    if(readImg())
        //meanThresholdTest(grayImg);
        ui->label_meanThreshold->setText("Threshold :" + QString::number(meanThreshold(grayImg) /255));
}
void Dialog::on_pushButton_clicked()
{
    if(readImg())
        //meanThresholdTest(grayImg);
        ui->label_meanThreshold->setText("Threshold :" + QString::number(meanStdThreshold(grayImg) /255));
}

bool Dialog::readImg()
{
    QString imgName = QFileDialog::getOpenFileName(this, "Open a file", QDir::currentPath().append("/images"));

    if(imgName.isEmpty())
        return false;
    qDebug() << imgName;
    origImg = cv::imread(imgName.toStdString());
    //grayImg = cv::imread(imgName.toStdString(), cv::IMREAD_GRAYSCALE);
    cv::cvtColor(origImg, grayImg, cv::COLOR_BGR2GRAY);

    QImage q_origImg(origImg.data, origImg.cols, origImg.rows, origImg.step, QImage::Format_BGR888);
    QImage q_grayImg(grayImg.data, grayImg.cols, grayImg.rows, grayImg.step, QImage::Format_Grayscale8);

    if(checkImgSize(q_origImg))
        q_origImg = q_origImg.scaled(700, 400, Qt::KeepAspectRatio);
    if(checkImgSize(q_grayImg))
        q_grayImg = q_grayImg.scaled(700, 400, Qt::KeepAspectRatio);

    // display orig image
    ui->label_orig->setPixmap(QPixmap::fromImage(q_origImg));
    // display gray image
    ui->label_gray->setPixmap(QPixmap::fromImage(q_grayImg));

    return true;
}

double Dialog::meanThreshold(cv::Mat src_Mat)
{
    // calculate Mean
    cv::Scalar tmp_mean;
    tmp_mean = cv::mean(src_Mat);
    qDebug() << tmp_mean.val[0] /255;

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
        cv::meanStdDev(dst_Mat1, tmp_mean1, tmp_stddev1, dst_Mat1 > 0);
        cv::meanStdDev(dst_Mat2, tmp_mean2, tmp_stddev2, dst_Mat2 > 0);
        // Tnext = (1/(s1+s2))*(s1*mean(fd(gd)) + s2*mean(fd(~gd)));
        // Tnext = (0.5)*(mean(fd(gd)) + mean(fd(~gd)));

        //qDebug() << calculateMean(dst_Mat1) /255;
        //thresholdNext = (tmp_mean1.val[0] + tmp_mean2.val[0]) /2;
        //thresholdNext = ((calculateMean(dst_Mat1) + calculateMean(dst_Mat2)))/2;
        thresholdNext = (tmp_mean1.val[0] + tmp_mean2.val[0]) /2;

        //thresholdNext = (1/ (tmp_stddev1.val[0] + tmp_stddev2.val[0]))  \
                * (tmp_stddev1.val[0] * tmp_mean1.val[0] + tmp_stddev2.val[0] * tmp_mean2.val[0]);

        done = (qAbs(threshold - thresholdNext) < 0.255);
        threshold = thresholdNext;
    }
    qDebug() << threshold /255;


    // dsplay
    cv::Mat noneDilationMat(src_Mat.rows, src_Mat.cols, src_Mat.type());
    cv::Mat dilatedMat(noneDilationMat.rows, noneDilationMat.cols, noneDilationMat.type());
    cv::threshold(src_Mat, noneDilationMat, threshold, 255, cv::THRESH_BINARY);

    cv::Mat kernel = kcircle();

    cv::morphologyEx(noneDilationMat, dilatedMat, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(dilatedMat, dilatedMat, cv::MORPH_OPEN, kernel);

    QImage q_dilatedMat(dilatedMat.data, dilatedMat.cols, dilatedMat.rows, dilatedMat.step, QImage::Format_Grayscale8);
    QImage q_noneDilationMat(noneDilationMat.data, noneDilationMat.cols, noneDilationMat.rows, noneDilationMat.step, QImage::Format_Grayscale8);

    if(checkImgSize(q_dilatedMat))
        q_dilatedMat = q_dilatedMat.scaled(700, 400, Qt::KeepAspectRatio);
    if(checkImgSize(q_noneDilationMat))
        q_noneDilationMat = q_noneDilationMat.scaled(700, 400, Qt::KeepAspectRatio);
    // display mean threshold image
    ui->label_meanstd->setPixmap(QPixmap::fromImage(q_dilatedMat));
    ui->label_mean->setPixmap(QPixmap::fromImage(q_noneDilationMat));


    /*
    QImage q_grayImg(dst_Mat1.data, dst_Mat1.cols, dst_Mat1.rows, dst_Mat1.step, QImage::Format_Grayscale8);
    QImage q_grayImg1(dst_Mat2.data, dst_Mat2.cols, dst_Mat2.rows, dst_Mat2.step, QImage::Format_Grayscale8);
    q_grayImg = q_grayImg.scaled(700, 400, Qt::KeepAspectRatio);
    q_grayImg1 = q_grayImg1.scaled(700, 400, Qt::KeepAspectRatio);
    ui->label_mean->setPixmap(QPixmap::fromImage(q_grayImg));
    ui->label_meanstd->setPixmap(QPixmap::fromImage(q_grayImg1));
    */

    return threshold;
}

double Dialog::meanStdThreshold(cv::Mat src_Mat)
{
    // calculate Mean
    cv::Scalar tmp_mean;
    tmp_mean = cv::mean(src_Mat);
    qDebug() << tmp_mean.val[0] /255;

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
        cv::meanStdDev(dst_Mat1, tmp_mean1, tmp_stddev1, dst_Mat1 > 0);
        cv::meanStdDev(dst_Mat2, tmp_mean2, tmp_stddev2, dst_Mat2 > 0);
        // Tnext = (1/(s1+s2))*(s1*mean(fd(gd)) + s2*mean(fd(~gd)));
        // Tnext = (0.5)*(mean(fd(gd)) + mean(fd(~gd)));

        //qDebug() << calculateMean(dst_Mat1) /255;
        //thresholdNext = (tmp_mean1.val[0] + tmp_mean2.val[0]) /2;
        //thresholdNext = ((calculateMean(dst_Mat1) + calculateMean(dst_Mat2)))/2;
        //thresholdNext = (tmp_mean1.val[0] + tmp_mean2.val[0]) /2;

        thresholdNext = (1/ (tmp_stddev1.val[0] + tmp_stddev2.val[0]))  \
                * (tmp_stddev1.val[0] * tmp_mean1.val[0] + tmp_stddev2.val[0] * tmp_mean2.val[0]);

        done = (qAbs(threshold - thresholdNext) < 0.255);
        threshold = thresholdNext;
    }
    qDebug() << threshold /255;


    // dsplay
    cv::Mat noneDilationMat(src_Mat.rows, src_Mat.cols, src_Mat.type());
    cv::Mat dilatedMat(noneDilationMat.rows, noneDilationMat.cols, noneDilationMat.type());
    cv::threshold(src_Mat, noneDilationMat, threshold, 255, cv::THRESH_BINARY);

    cv::Mat kernel = kcircle();

    cv::morphologyEx(noneDilationMat, dilatedMat, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(dilatedMat, dilatedMat, cv::MORPH_OPEN, kernel);

    QImage q_dilatedMat(dilatedMat.data, dilatedMat.cols, dilatedMat.rows, dilatedMat.step, QImage::Format_Grayscale8);
    QImage q_noneDilationMat(noneDilationMat.data, noneDilationMat.cols, noneDilationMat.rows, noneDilationMat.step, QImage::Format_Grayscale8);

    if(checkImgSize(q_dilatedMat))
        q_dilatedMat = q_dilatedMat.scaled(700, 400, Qt::KeepAspectRatio);
    if(checkImgSize(q_noneDilationMat))
        q_noneDilationMat = q_noneDilationMat.scaled(700, 400, Qt::KeepAspectRatio);
    // display mean threshold image
    ui->label_meanstd->setPixmap(QPixmap::fromImage(q_dilatedMat));
    ui->label_mean->setPixmap(QPixmap::fromImage(q_noneDilationMat));


    /*
    QImage q_grayImg(dst_Mat1.data, dst_Mat1.cols, dst_Mat1.rows, dst_Mat1.step, QImage::Format_Grayscale8);
    QImage q_grayImg1(dst_Mat2.data, dst_Mat2.cols, dst_Mat2.rows, dst_Mat2.step, QImage::Format_Grayscale8);
    q_grayImg = q_grayImg.scaled(700, 400, Qt::KeepAspectRatio);
    q_grayImg1 = q_grayImg1.scaled(700, 400, Qt::KeepAspectRatio);
    ui->label_mean->setPixmap(QPixmap::fromImage(q_grayImg));
    ui->label_meanstd->setPixmap(QPixmap::fromImage(q_grayImg1));
    */

    return threshold;
}

cv::Mat Dialog::kcircle()
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    *kernel.ptr(2, 0) = 0;
    *kernel.ptr(1, 1) = 0;
    *kernel.ptr(1, 5) = 0;
    *kernel.ptr(4, 0) = 0;
    *kernel.ptr(2, 6) = 0;
    *kernel.ptr(4, 6) = 0;
    *kernel.ptr(5, 1) = 0;
    *kernel.ptr(5, 5) = 0;
    return kernel;
}
/*
double Dialog::calculateMean(cv::Mat src_Mat)
{
    int nChannels = src_Mat.channels();
    int nRows = src_Mat.rows;
    int nCols = src_Mat.cols * nChannels;
    int nStep = src_Mat.step;

    double sum;
    int count = 0;
    uchar* srcData= src_Mat.data;
    for( int j = 0; j < nRows; j++ ){
        for( int i = 0; i < nCols; i++ ) {
            if(*src_Mat.ptr(j, i) != 0)
            {
                sum += *src_Mat.ptr(j, i);
                count++;
            }
        }
        srcData += nStep;
    }
    return sum / count;
}
*/


bool Dialog::checkImgSize(QImage src_Img)
{
    return (src_Img.width() > 700);
}


