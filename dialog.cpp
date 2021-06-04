#include "dialog.h"
#include "ui_dialog.h"
#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"
#include <opencv2/core.hpp>

#include "adjustthreshold.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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

    cv::Mat dst2;
    labels.convertTo(dst2, CV_8U);
/*
    for(int j = 0; j < dst2.cols; j++)
        for(int i = 0; i < dst2.rows; i++)
        {
            int pixelLabel = *dst2.ptr(j, i);

            if(stats.at<int>(pixelLabel, cv::CC_STAT_AREA) < 8000)
                *dst2.ptr(j, i) = 0;
        }
*/

    for(int label = 1; label < nLabels; ++label){
        //colors[label] = cv::Vec3b( (std::rand()&255), (std::rand()&255), (std::rand()&255) );
        std::cout << "Component "<< label << std::endl;
        std::cout << "CC_STAT_LEFT   = " << stats.at<int>(label,cv::CC_STAT_LEFT) << std::endl;
        std::cout << "CC_STAT_TOP    = " << stats.at<int>(label,cv::CC_STAT_TOP) << std::endl;
        std::cout << "CC_STAT_WIDTH  = " << stats.at<int>(label,cv::CC_STAT_WIDTH) << std::endl;
        std::cout << "CC_STAT_HEIGHT = " << stats.at<int>(label,cv::CC_STAT_HEIGHT) << std::endl;
        std::cout << "CC_STAT_AREA   = " << stats.at<int>(label,cv::CC_STAT_AREA) << std::endl;
        std::cout << "CENTER   = (" << centroids.at<double>(label, 0) <<","<< centroids.at<double>(label, 1) << ")"<< std::endl << std::endl;
    }

    cv::Mat dst(grayImg.size(), CV_8U);


    /*
    // older usage
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){covarianceMatrix
            int label = labels.at<int>(r, c);
            //::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
            *dst.ptr(r, c) = label;
            //pixel = colors[label];
        }
    }
    */

    int whichLabel = 4;
    cv::Mat tmp2 = (dst2 == whichLabel);
    cv::Mat outputImg1;
    cv::cvtColor(tmp2, outputImg1, cv::COLOR_GRAY2BGR);

    cv::Mat outputImg2;
    outputImg2 = dst2*50;

    cv::Mat mean, eigenvectors, eigenvalues;



    int64 t0 = cv::getTickCount();
/***************************************/

    cv::Moments imgMoment;
    imgMoment = cv::moments(tmp2);

    Eigen::Matrix<double, 2, 2> covarianceMatrix;
    covarianceMatrix << imgMoment.mu20, imgMoment.mu11,
                        imgMoment.mu11, imgMoment.mu02 ;

/***************************************/
    int64 t1 = cv::getTickCount();
    double t = (t1-t0) * 1000 /cv::getTickFrequency();
    qDebug() << "Calculate moment times: " << t <<"ms";

    double minEigen = covarianceMatrix.eigenvalues().real()[1];
    double maxEigen = covarianceMatrix.eigenvalues().real()[0];
    qDebug() << minEigen << maxEigen;

    double major = 2 * sqrt(maxEigen/ imgMoment.m00);
    double minor = 2 * sqrt(minEigen/ imgMoment.m00);

    double theta = 0.5 * atan2(2*imgMoment.mu11, imgMoment.mu20 - imgMoment.mu02);

    cv::Point centerPoint(centroids.at<double>(whichLabel, 0), centroids.at<double>(whichLabel, 1));
    cv::Size axisSize(major, minor);
    qDebug() << major << minor << qRadiansToDegrees(theta);

    cv::ellipse(outputImg1, centerPoint, axisSize, qRadiansToDegrees(theta), 0, 360, cv::Scalar(0, 0, 255), 2);


    std::vector<cv::Point> featurePoint;
    cv::Point featurePoint1 = cv::Point(major*cos(theta), major*sin(theta)) + centerPoint;
    cv::Point featurePoint2 = cv::Point(minor*sin(M_PI - theta), minor*cos(M_PI - theta)) + centerPoint;
    cv::Point featurePoint3 = cv::Point(major*cos(theta + M_PI), major*sin(theta + M_PI)) + centerPoint;
    cv::Point featurePoint4 = cv::Point(minor*sin(-theta), minor*cos(-theta)) + centerPoint;
    featurePoint.push_back(centerPoint);
    featurePoint.push_back(featurePoint1);
    featurePoint.push_back(featurePoint2);
    featurePoint.push_back(featurePoint3);
    featurePoint.push_back(featurePoint4);
    std::cout << featurePoint[2] << std::endl;

    //qDebug() << featurePoint.pop_back();

    //std::cout << featurePoint1 << featurePoint2 << featurePoint3 << featurePoint4<< std::endl;

    cv::circle(outputImg1, centerPoint, 1, cv::Scalar(0, 0, 255), -1);
    cv::circle(outputImg1, featurePoint1, 5, cv::Scalar(255, 0, 0), -1);
    cv::circle(outputImg1, featurePoint2, 5, cv::Scalar(255, 0, 0), -1);
    cv::circle(outputImg1, featurePoint3, 5, cv::Scalar(255, 0, 0), -1);
    cv::circle(outputImg1, featurePoint4, 5, cv::Scalar(255, 0, 0), -1);

    imageDisplay(origImg, tmp2, outputImg1, outputImg2);

}

void Dialog::on_pushButton_clicked()
{


    Eigen::Matrix<double, 10, 10> B;
    Eigen::MatrixXd::Identity(10,10);
    B.setIdentity(10,10);

    std::cout << B << std::endl;

    std::cout << B.eigenvalues() << std::endl;
    Eigen::Matrix<double, 2, 2> A;
    A(1, 1) = 1;
    qDebug() << A(1, 1);

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
    QImage q_imgMat3(imgMat3.data, imgMat3.cols, imgMat3.rows, imgMat3.step, QImage::Format_BGR888);
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
