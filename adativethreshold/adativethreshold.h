#ifndef ADATIVETHRESHOLD_H
#define ADATIVETHRESHOLD_H

#include "adativethreshold_global.h"
#include <QtCore>
#include "opencv2/opencv.hpp"

enum ThresholdMethod
{
    THRESHOLD_MEAN      = 1,
    THRESHOLD_MEANstd   = 2,
    THRESHOLD_OTSU      = 3
};
class ADATIVETHRESHOLD_EXPORT Adativethreshold
{
public:
    Adativethreshold();

    bool readImg(std::string);

    double getGlobalAdativeThreshold(int, double);

    double getGlobalAdativeThreshold(cv::Mat, int, double);

    cv::Mat getNiblackThreshold(cv::Mat _src, cv::Mat _dst, double maxValue,
                                int type, int blockSize, double k, int binarizationMethod, double r = 128);

    cv::Mat kcircle(int);

    cv::Mat morphologyClosingOpening(cv::Mat, int);

    cv::Mat getOrigMat();

    cv::Mat getGrayMat();
private:
    cv::Mat origImg;
    cv::Mat grayImg;

    bool imgExist = false;

};

#endif // ADATIVETHRESHOLD_H
