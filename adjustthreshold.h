#ifndef ADJUSTTHRESHOLD_H
#define ADJUSTTHRESHOLD_H

#include "adjustthreshold_global.h"
#include <QtCore>
#include "opencv2/opencv.hpp"

enum ThresholdMethod
{
    THRESHOLD_MEAN      = 1,
    THRESHOLD_MEANstd   = 2,
    THRESHOLD_OTSU      = 3
};

class ADJUSTTHRESHOLD_EXPORT Adjustthreshold
{
public:
    Adjustthreshold();

    bool readImg(std::string);

    double getAdjustThreshold(int, double);

    double getAdjustThreshold(cv::Mat, int, double);

    cv::Mat kcircle(int);

    cv::Mat morphologyClosingOpening(cv::Mat, int);

    cv::Mat getOrigMat();

    cv::Mat getGrayMat();
private:
    cv::Mat origImg;
    cv::Mat grayImg;

    bool imgExist = false;

};

#endif // ADJUSTTHRESHOLD_H
