#include "dialog.h"
#include "ui_dialog.h"
#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"


#include "adjustthreshold.h"
#include <QFileDialog>
#include <QDebug>

#define ThresholdError 0.255

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
    int64 t0 = cv::getTickCount();

    if(readImg())
        ui->label_meanThreshold->setText("Threshold :" + QString::number(meanThreshold(grayImg) /255));

    int64 t1 = cv::getTickCount();
    double t = (t1-t0) * 1000 /cv::getTickFrequency();
    qDebug() << "Detecting time on a single frame: " << t <<"ms";
}

void Dialog::on_pushButton_clicked()
{
    if(readImg())
        ui->label_meanThreshold->setText("Threshold :" + QString::number(meanStdThreshold(grayImg) /255));
}

bool Dialog::readImg()
{
    QString imgName = QFileDialog::getOpenFileName(this, "Open a file", QDir::currentPath().append("/images"));

    if(imgName.isEmpty())
        return false;

    qDebug() << imgName;
    origImg = cv::imread(imgName.toStdString());
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

    // create two Mat use to calculate each group mean & std
    cv::Mat dst_Mat1(src_Mat.rows, src_Mat.cols, src_Mat.type());
    cv::Mat dst_Mat2(src_Mat.rows, src_Mat.cols, src_Mat.type());

    while(!done)
    {
        cv::threshold(src_Mat, dst_Mat1, threshold, 255, cv::THRESH_TOZERO);
        cv::threshold(src_Mat, dst_Mat2, threshold, 255, cv::THRESH_TOZERO_INV);

        cv::Scalar tmp_mean1, tmp_stddev1, tmp_mean2, tmp_stddev2;

        // calculate two group' mean & std except 0
        cv::meanStdDev(dst_Mat1, tmp_mean1, tmp_stddev1, src_Mat > threshold);
        cv::meanStdDev(dst_Mat2, tmp_mean2, tmp_stddev2, src_Mat < threshold);

        // Tnext = (1/(s1+s2))*(s1*mean(fd(gd)) + s2*mean(fd(~gd)));
        // Tnext = (0.5)*(mean(fd(gd)) + mean(fd(~gd)));
        thresholdNext = (tmp_mean1.val[0] + tmp_mean2.val[0]) /2;

        //thresholdNext = (1/ (tmp_stddev1.val[0] + tmp_stddev2.val[0]))  \
                * (tmp_stddev1.val[0] * tmp_mean1.val[0] + tmp_stddev2.val[0] * tmp_mean2.val[0]);

        done = (qAbs(threshold - thresholdNext) < ThresholdError);
        threshold = thresholdNext;
    }
    qDebug() << threshold /255;


    // display
    cv::Mat noneDilationMat(src_Mat.rows, src_Mat.cols, src_Mat.type());
    cv::Mat dilatedMat(noneDilationMat.rows, noneDilationMat.cols, noneDilationMat.type());
    cv::threshold(src_Mat, noneDilationMat, threshold, 255, cv::THRESH_BINARY);

    dilatedMat = morphologyClosingOpening(noneDilationMat, 3);

    // change cv::Mat to QImgae to display
    QImage q_dilatedMat(dilatedMat.data, dilatedMat.cols, dilatedMat.rows, dilatedMat.step, QImage::Format_Grayscale8);
    QImage q_noneDilationMat(noneDilationMat.data, noneDilationMat.cols, noneDilationMat.rows, noneDilationMat.step, QImage::Format_Grayscale8);

    // check image size
    if(checkImgSize(q_dilatedMat))
        q_dilatedMat = q_dilatedMat.scaled(700, 400, Qt::KeepAspectRatio);
    if(checkImgSize(q_noneDilationMat))
        q_noneDilationMat = q_noneDilationMat.scaled(700, 400, Qt::KeepAspectRatio);
    // display image
    ui->label_meanstd->setPixmap(QPixmap::fromImage(q_dilatedMat));
    ui->label_mean->setPixmap(QPixmap::fromImage(q_noneDilationMat));

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

    // create two Mat use to calculate each group mean & std
    cv::Mat dst_Mat1(src_Mat.rows, src_Mat.cols, src_Mat.type());
    cv::Mat dst_Mat2(src_Mat.rows, src_Mat.cols, src_Mat.type());

    while(!done)
    {
        cv::threshold(src_Mat, dst_Mat1, threshold, 255, cv::THRESH_TOZERO);
        cv::threshold(src_Mat, dst_Mat2, threshold, 255, cv::THRESH_TOZERO_INV);

        cv::Scalar tmp_mean1, tmp_stddev1, tmp_mean2, tmp_stddev2;

        // calculate two group' mean & std except 0
        cv::meanStdDev(dst_Mat1, tmp_mean1, tmp_stddev1, dst_Mat1 > 0);
        cv::meanStdDev(dst_Mat2, tmp_mean2, tmp_stddev2, dst_Mat2 > 0);

        // Tnext = (1/(s1+s2))*(s1*mean(fd(gd)) + s2*mean(fd(~gd)));
        // Tnext = (0.5)*(mean(fd(gd)) + mean(fd(~gd)));
        thresholdNext = (1/ (tmp_stddev1.val[0] + tmp_stddev2.val[0]))  \
                * (tmp_stddev1.val[0] * tmp_mean1.val[0] + tmp_stddev2.val[0] * tmp_mean2.val[0]);

        done = (qAbs(threshold - thresholdNext) < ThresholdError);
        threshold = thresholdNext;
    }
    qDebug() << threshold /255;

    // dsplay
    cv::Mat noneDilationMat(src_Mat.rows, src_Mat.cols, src_Mat.type());
    cv::Mat dilatedMat(noneDilationMat.rows, noneDilationMat.cols, noneDilationMat.type());

    cv::threshold(src_Mat, noneDilationMat, threshold, 255, cv::THRESH_BINARY);

    dilatedMat = morphologyClosingOpening(noneDilationMat, 3);
    // change cv::Mat to QImgae to display
    QImage q_dilatedMat(dilatedMat.data, dilatedMat.cols, dilatedMat.rows, dilatedMat.step, QImage::Format_Grayscale8);
    QImage q_noneDilationMat(noneDilationMat.data, noneDilationMat.cols, noneDilationMat.rows, noneDilationMat.step, QImage::Format_Grayscale8);

    // check image size
    if(checkImgSize(q_dilatedMat))
        q_dilatedMat = q_dilatedMat.scaled(700, 400, Qt::KeepAspectRatio);
    if(checkImgSize(q_noneDilationMat))
        q_noneDilationMat = q_noneDilationMat.scaled(700, 400, Qt::KeepAspectRatio);
    // display image
    ui->label_meanstd->setPixmap(QPixmap::fromImage(q_dilatedMat));
    ui->label_mean->setPixmap(QPixmap::fromImage(q_noneDilationMat));

    return threshold;
}

cv::Mat Dialog::kcircle(int kCircleRadius)
{
    int kernelSize = kCircleRadius *2 + 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(kernelSize, kernelSize));

    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;

    for(int j = 0; j < kernelCols; j++)
        for(int i = 0; i < kernelRows; i++)
        {
            int colDistance = qAbs(j - kCircleRadius);
            int rowDistance = qAbs(i - kCircleRadius);

            if((colDistance + rowDistance) <= kCircleRadius)
                *kernel.ptr(j, i) = 1;
            else
                *kernel.ptr(j, i) = 0;
        }

    return kernel;
}

cv::Mat Dialog::morphologyClosingOpening(cv::Mat src_Mat, int kCircleRadius)
{
    cv::Mat dst_Mat(src_Mat.rows, src_Mat.cols, src_Mat.type());
    // create kernel for dilation
    cv::Mat kernel = kcircle(kCircleRadius);

    // closing operate
    cv::morphologyEx(src_Mat, dst_Mat, cv::MORPH_CLOSE, kernel);
    // opening operate
    cv::morphologyEx(dst_Mat, dst_Mat, cv::MORPH_OPEN, kernel);

    return dst_Mat;
}

bool Dialog::checkImgSize(QImage src_Img)
{
    return (src_Img.width() > 700);
}

void Dialog::on_pushButton_2_clicked()
{
    // mean threshold
    QString imgName = QFileDialog::getOpenFileName(this, "Open a file", QDir::currentPath().append("/images"));

    Adjustthreshold mThreshold;

    mThreshold.readImg(imgName.toStdString());
    origImg = mThreshold.getOrigMat();
    grayImg = mThreshold.getGrayMat();

    qDebug() << mThreshold.getAdjustThreshold(THRESHOLD_MEAN, 0.255)/255;
    double thresholdMean = mThreshold.getAdjustThreshold(THRESHOLD_MEAN, 0.255);

    cv::Mat outputImg1(grayImg.rows, grayImg.cols, grayImg.type());
    cv::Mat outputImg2(grayImg.rows, grayImg.cols, grayImg.type());

    cv::threshold(grayImg, outputImg1, thresholdMean, 255, cv::THRESH_BINARY);
    outputImg2 = mThreshold.morphologyClosingOpening(outputImg1, 3);

    imageDisplay(origImg, grayImg, outputImg1, outputImg2);

    //mlib.test();
}

void Dialog::on_pushButton_3_clicked()
{
    // mean std threshold
    Adjustthreshold mThreshold;
    QString imgName = QFileDialog::getOpenFileName(this, "Open a file", QDir::currentPath().append("/images"));
    if(!imgName.isEmpty())
    {
        mThreshold.readImg(imgName.toStdString());
        origImg = mThreshold.getOrigMat();
        grayImg = mThreshold.getGrayMat();
        cv::Mat outputImg1(grayImg.rows, grayImg.cols, grayImg.type());
        cv::Mat outputImg2(grayImg.rows, grayImg.cols, grayImg.type());

        //cv::ximgproc::niBlackThreshold(grayImg, outputImg1)
        cv::adaptiveThreshold(grayImg, outputImg1, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 101, -10);
        cv::adaptiveThreshold(grayImg, outputImg2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 101, -20);

        //cv::ximgproc::niBlackThreshold(grayImg, outputImg2, 255, cv::THRESH_BINARY, 101, 0.6, cv::ximgproc::BINARIZATION_NIBLACK);
        //outputImg2 = mThreshold.morphologyClosingOpening(outputImg1, 3);

        imageDisplay(outputImg1, outputImg2);
    }
}

void Dialog::imageDisplay(cv::Mat imgMat1, cv::Mat imgMat2, cv::Mat imgMat3, cv::Mat imgMat4)
{
    QImage q_imgMat1(imgMat1.data, imgMat1.cols, imgMat1.rows, imgMat1.step, QImage::Format_Grayscale8);
    QImage q_imgMat2(imgMat2.data, imgMat2.cols, imgMat2.rows, imgMat2.step, QImage::Format_Grayscale8);
    QImage q_imgMat3(imgMat3.data, imgMat3.cols, imgMat3.rows, imgMat3.step, QImage::Format_Grayscale8);
    QImage q_imgMat4(imgMat4.data, imgMat4.cols, imgMat4.rows, imgMat4.step, QImage::Format_Grayscale8);

    // check image size
    if(checkImgSize(q_imgMat1))
        q_imgMat1 = q_imgMat1.scaled(700, 400, Qt::KeepAspectRatio);

    if(checkImgSize(q_imgMat2))
        q_imgMat2 = q_imgMat2.scaled(700, 400, Qt::KeepAspectRatio);

    if(checkImgSize(q_imgMat3))
        q_imgMat3 = q_imgMat3.scaled(700, 400, Qt::KeepAspectRatio);

    if(checkImgSize(q_imgMat4))
        q_imgMat4 = q_imgMat4.scaled(700, 400, Qt::KeepAspectRatio);

    // display image
    ui->label_orig->setPixmap(QPixmap::fromImage(q_imgMat1));
    ui->label_gray->setPixmap(QPixmap::fromImage(q_imgMat2));
    ui->label_mean->setPixmap(QPixmap::fromImage(q_imgMat3));
    ui->label_meanstd->setPixmap(QPixmap::fromImage(q_imgMat4));

}

void Dialog::imageDisplay(cv::Mat imgMat1, cv::Mat imgMat2)
{
    QImage q_imgMat1(imgMat1.data, imgMat1.cols, imgMat1.rows, imgMat1.step, QImage::Format_Grayscale8);
    QImage q_imgMat2(imgMat2.data, imgMat2.cols, imgMat2.rows, imgMat2.step, QImage::Format_Grayscale8);
    if(checkImgSize(q_imgMat1))
        q_imgMat1 = q_imgMat1.scaled(700, 400, Qt::KeepAspectRatio);

    if(checkImgSize(q_imgMat2))
        q_imgMat2 = q_imgMat2.scaled(700, 400, Qt::KeepAspectRatio);

    ui->label_orig->setPixmap(QPixmap::fromImage(q_imgMat1));
    ui->label_gray->setPixmap(QPixmap::fromImage(q_imgMat2));

}
