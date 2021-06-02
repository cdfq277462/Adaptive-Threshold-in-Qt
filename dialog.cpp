#include "dialog.h"
#include "ui_dialog.h"
#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
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
    QString imgName = QFileDialog::getOpenFileName(this, "Open a file", QDir::currentPath().append("/images"));

    Adjustthreshold mThreshold;

    mThreshold.readImg(imgName.toStdString());
    origImg = mThreshold.getOrigMat();
    grayImg = mThreshold.getGrayMat();
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;


    int nLabels = cv::connectedComponentsWithStats(grayImg, labels, stats, centroids, 8, CV_32S);

    qDebug() << labels.rows << labels.cols;
    qDebug() << stats.rows << stats.cols;
    qDebug() << centroids.rows << centroids.cols;

    std::vector<cv::Vec3b> colors(nLabels);
    colors[0] = cv::Vec3b(0, 0, 0);


    for(int label = 1; label < nLabels; ++label){
        colors[label] = cv::Vec3b( (std::rand()&255), (std::rand()&255), (std::rand()&255) );
        std::cout << "Component "<< label << std::endl;
        std::cout << "CC_STAT_LEFT   = " << stats.at<int>(label,cv::CC_STAT_LEFT) << std::endl;
        std::cout << "CC_STAT_TOP    = " << stats.at<int>(label,cv::CC_STAT_TOP) << std::endl;
        std::cout << "CC_STAT_WIDTH  = " << stats.at<int>(label,cv::CC_STAT_WIDTH) << std::endl;
        std::cout << "CC_STAT_HEIGHT = " << stats.at<int>(label,cv::CC_STAT_HEIGHT) << std::endl;
        std::cout << "CC_STAT_AREA   = " << stats.at<int>(label,cv::CC_STAT_AREA) << std::endl;
        std::cout << "CENTER   = (" << centroids.at<double>(label, 0) <<","<< centroids.at<double>(label, 1) << ")"<< std::endl << std::endl;
    }

    cv::Mat dst(grayImg.size(), CV_8U);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labels.at<int>(r, c);
            //::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
            *dst.ptr(r, c) = label;
            //pixel = colors[label];
        }
    }
    //cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
    qDebug() << labels.at<int>(245, 425);
    qDebug() << labels.at<int>(425, 245);
    cv::Mat tmp2 = (dst == 2);
    cv::Mat tmp3 = (dst == 3);
    cv::Mat tmp4 = (dst == 4);


    cv::Moments imgMoment;
    imgMoment = cv::moments(dst);
    qDebug() << imgMoment.m00 << imgMoment.m01 << imgMoment.m10;
    qDebug() << imgMoment.m10 / imgMoment.m00;
    imageDisplay(grayImg, tmp2, tmp3, tmp4);
}

void Dialog::on_pushButton_clicked()
{

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
    return (src_Img.width() > 300);
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
        cv::Mat tmp(grayImg.rows, grayImg.cols, grayImg.type());

        //cv::ximgproc::niBlackThreshold(grayImg, outputImg1)

        //cv::ximgproc::niBlackThreshold(grayImg, outputImg2, 255, cv::THRESH_BINARY, 101, -0.6, cv::ximgproc::BINARIZATION_NIBLACK);
        tmp = niBlackThreshold_custom(grayImg, outputImg1, 255, cv::THRESH_BINARY, 201, -0.6, cv::ximgproc::BINARIZATION_NIBLACK);
        tmp = tmp *1.3;

        //imageDisplay(tmp, outputImg1);

        double thresholdGlobal, threshold;
        double thresholdNext    = 0;
        double thresholdError   = 0.255;
        double thresholdGain    = 1.3;
        threshold = cv::mean(tmp).val[0];
        thresholdGlobal = threshold;
        bool done = (abs(threshold - thresholdNext) < thresholdError);
        cv::Scalar tmp_mean1, tmp_stddev1, tmp_mean2, tmp_stddev2;

        while(!done)
        {
            tmp_mean1 = cv::mean(grayImg, grayImg > threshold) ;
            tmp_mean2 = cv::mean(grayImg, grayImg < threshold) ;
            thresholdNext = (tmp_mean1.val[0] + tmp_mean2.val[0]) /2;

            done = (abs(threshold - thresholdNext) < thresholdError);
            threshold = thresholdNext;
        }
        qDebug() << "threshold: " << threshold << "thresholdGlobal: " << thresholdGlobal;

        cv::threshold(grayImg, outputImg1, threshold, 255, cv::THRESH_BINARY);

        outputImg2 = grayImg >= tmp;
        cv::bitwise_and(outputImg1, outputImg2, outputImg2);

        outputImg2 = mThreshold.morphologyClosingOpening(outputImg2, 3);
        imageDisplay(origImg, tmp, outputImg1, outputImg2);
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

cv::Mat Dialog::niBlackThreshold_custom(cv::Mat _src, cv::Mat _dst, double maxValue,
                                        int type, int blockSize, double k, int binarizationMethod, double r)
{
    // Input grayscale image
    cv::Mat src = _src;
    type &= cv::THRESH_MASK;

    // Compute local threshold (T = mean + k * stddev)
    // using mean and standard deviation in the neighborhood of each pixel
    // (intermediate calculations are done with floating-point precision)
    cv::Mat thresh;
    {
        // note that: Var[X] = E[X^2] - E[X]^2
        cv::Mat mean, sqmean, variance, stddev, sqrtVarianceMeanSum;
        double srcMin, stddevMax;
        cv::boxFilter(src, mean, CV_32F, cv::Size(blockSize, blockSize),
                cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
        cv::sqrBoxFilter(src, sqmean, CV_32F, cv::Size(blockSize, blockSize),
                cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
        variance = sqmean - mean.mul(mean);
        sqrt(variance, stddev);
        switch (binarizationMethod)
        {
        case cv::ximgproc::BINARIZATION_NIBLACK:
            thresh = mean + stddev * static_cast<float>(k);
            break;
        case cv::ximgproc::BINARIZATION_SAUVOLA:
            thresh = mean.mul(1. + static_cast<float>(k) * (stddev / r - 1.));
            break;
        case cv::ximgproc::BINARIZATION_WOLF:
            minMaxIdx(src, &srcMin);
            minMaxIdx(stddev, NULL, &stddevMax);
            thresh = mean - static_cast<float>(k) * (mean - srcMin - stddev.mul(mean - srcMin) / stddevMax);
            break;
        case cv::ximgproc::BINARIZATION_NICK:
            sqrt(variance + sqmean, sqrtVarianceMeanSum);
            thresh = mean + static_cast<float>(k) * sqrtVarianceMeanSum;
            break;
        default:
            //cv::CV_Error(cv::CV_StsBadArg, "Unknown binarization method");
            break;
        }
        thresh.convertTo(thresh, src.depth());
    }
    return thresh;
    // Prepare output image
    _dst.create(src.size(), src.type());
    cv::Mat dst = _dst;
    //cv::CV_Assert(src.data != dst.data);  // no inplace processing

    // Apply thresholding: ( pixel > threshold ) ? foreground : background
    cv::Mat mask;
    switch (type)
    {
    case cv::THRESH_BINARY:      // dst = (src > thresh) ? maxval : 0
    case cv::THRESH_BINARY_INV:  // dst = (src > thresh) ? 0 : maxval
        compare(src, thresh, mask, (type == cv::THRESH_BINARY ? cv::CMP_GT : cv::CMP_LE));
        dst.setTo(0);
        dst.setTo(maxValue, mask);
        break;
    case cv::THRESH_TRUNC:       // dst = (src > thresh) ? thresh : src
        compare(src, thresh, mask, cv::CMP_GT);
        src.copyTo(dst);
        thresh.copyTo(dst, mask);
        break;
    case cv::THRESH_TOZERO:      // dst = (src > thresh) ? src : 0
    case cv::THRESH_TOZERO_INV:  // dst = (src > thresh) ? 0 : src
        cv::compare(src, thresh, mask, (type == cv::THRESH_TOZERO ? cv::CMP_GT : cv::CMP_LE));
        dst.setTo(0);
        src.copyTo(dst, mask);
        break;
    default:
        //cv::CV_Error( cv::CV_StsBadArg, "Unknown threshold type" );
        break;
    }

}
