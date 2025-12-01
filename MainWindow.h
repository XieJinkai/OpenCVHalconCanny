#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollArea>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLoadImage();
    void onProcessImage();

private:
    void setupUi();
    void processOpenCV(const QString &inputPath, int low, int high, int blurK);
    void processHalcon(const QString &inputPath, int low, int high, int blurK);
    void displayImage(const QString &path);

    QWidget *centralWidget;
    QLabel *imageLabel;
    QScrollArea *scrollArea;
    
    QPushButton *btnLoad;
    QPushButton *btnProcess;
    
    QComboBox *comboBackend;
    QSpinBox *spinLow;
    QSpinBox *spinHigh;
    QSpinBox *spinBlur;
    
    QString currentImagePath;
};

#endif // MAINWINDOW_H
