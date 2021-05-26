#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include <QtCore>
#include <QtGui>
#include <QGraphicsScene>
#include "opencv2/opencv.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class Dialog; }
QT_END_NAMESPACE

class Dialog : public QDialog
{
    Q_OBJECT

public:
    Dialog(QWidget *parent = nullptr);
    ~Dialog();

private slots:
    void on_pushButton_open_clicked();

    bool readImg();

    double meanThreshold(cv::Mat);

    double meanStdThreshold(cv::Mat);

    // useless
    // double calculateMean(cv::Mat);

    cv::Mat kcircle(int);

    cv::Mat morphologyClosingOpening(cv::Mat);

    void on_pushButton_clicked();

    bool checkImgSize(QImage);

private:
    Ui::Dialog *ui;
    QGraphicsScene *imageScene;

    cv::Mat origImg, grayImg;
};
#endif // DIALOG_H
