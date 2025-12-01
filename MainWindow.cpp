#include "MainWindow.h"
#include <QDebug>
#include <QFileInfo>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include <halconcpp/HalconCpp.h>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUi();
    resize(1000, 800);
    setWindowTitle("OpenCV & Halcon Canny Edge Detection v2.1");
}

MainWindow::~MainWindow() {
}

void MainWindow::setupUi() {
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // Controls Area
    QHBoxLayout *controlsLayout = new QHBoxLayout();
    
    btnLoad = new QPushButton("加载图像", this);
    controlsLayout->addWidget(btnLoad);

    controlsLayout->addWidget(new QLabel("后端:", this));
    comboBackend = new QComboBox(this);
    comboBackend->addItem("OpenCV");
    comboBackend->addItem("Halcon");
    controlsLayout->addWidget(comboBackend);

    controlsLayout->addWidget(new QLabel("低阈值:", this));
    spinLow = new QSpinBox(this);
    spinLow->setRange(0, 255);
    spinLow->setValue(70);
    controlsLayout->addWidget(spinLow);

    controlsLayout->addWidget(new QLabel("高阈值:", this));
    spinHigh = new QSpinBox(this);
    spinHigh->setRange(0, 255);
    spinHigh->setValue(150);
    controlsLayout->addWidget(spinHigh);

    controlsLayout->addWidget(new QLabel("模糊核:", this));
    spinBlur = new QSpinBox(this);
    spinBlur->setRange(1, 31);
    spinBlur->setSingleStep(2);
    spinBlur->setValue(5);
    controlsLayout->addWidget(spinBlur);

    btnProcess = new QPushButton("处理", this);
    controlsLayout->addWidget(btnProcess);
    
    controlsLayout->addStretch();

    mainLayout->addLayout(controlsLayout);

    // Image Display Area
    scrollArea = new QScrollArea(this);
    imageLabel = new QLabel(this);
    imageLabel->setAlignment(Qt::AlignCenter);
    scrollArea->setWidget(imageLabel);
    scrollArea->setWidgetResizable(true);
    mainLayout->addWidget(scrollArea);

    // Connections
    connect(btnLoad, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    connect(btnProcess, &QPushButton::clicked, this, &MainWindow::onProcessImage);
}

void MainWindow::onLoadImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "打开图像", "", "Images (*.png *.jpg *.bmp *.jpeg)");
    if (!fileName.isEmpty()) {
        currentImagePath = fileName;
        displayImage(currentImagePath);
    }
}

void MainWindow::displayImage(const QString &path) {
    QPixmap pixmap(path);
    if (!pixmap.isNull()) {
        imageLabel->setPixmap(pixmap);
        imageLabel->adjustSize();
    } else {
        QMessageBox::warning(this, "错误", "无法加载图像进行显示。");
    }
}

void MainWindow::onProcessImage() {
    if (currentImagePath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先加载图像。");
        return;
    }

    int low = spinLow->value();
    int high = spinHigh->value();
    int blurK = spinBlur->value();
    if (blurK % 2 == 0) blurK++; // Ensure odd kernel size

    QString backend = comboBackend->currentText();
    
    if (backend == "OpenCV") {
        processOpenCV(currentImagePath, low, high, blurK);
    } else {
        processHalcon(currentImagePath, low, high, blurK);
    }
}

void MainWindow::processOpenCV(const QString &inputPath, int low, int high, int blurK) {
#ifndef HAVE_OPENCV
    QMessageBox::critical(this, "错误", "OpenCV 后端不可用 (HAVE_OPENCV 未定义)。");
#else
    std::string input = inputPath.toStdString();
    cv::Mat src = cv::imread(input, cv::IMREAD_COLOR);
    if (src.empty()) {
        QMessageBox::critical(this, "错误", "OpenCV 读取图像失败。");
        return;
    }

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    if (blurK > 1) {
        cv::GaussianBlur(gray, gray, cv::Size(blurK, blurK), blurK * 0.2);
    }
    
    cv::Mat edges;
    cv::Canny(gray, edges, low, high);
    
    std::string output = "temp_result_opencv.png";
    cv::imwrite(output, edges);
    
    displayImage(QString::fromStdString(output));
#endif
}

void MainWindow::processHalcon(const QString &inputPath, int low, int high, int blurK) {
    try {
        using namespace HalconCpp;
        // Use Local8Bit for Windows paths to ensure Chinese characters are handled correctly by Halcon default
        std::string input = inputPath.toLocal8Bit().constData();
        
        HImage himg;
        ReadImage(&himg, HTuple(input.c_str()));
        
        // Check if image was loaded
        HTuple count;
        CountObj(himg, &count);
        if (count.I() == 0) {
            QMessageBox::warning(this, "警告", "Halcon 未能加载图像 (对象为空)。");
            return;
        }
        
        // If multiple images, select the first one
        HImage workingImg;
        if (count.I() > 1) {
            SelectObj(himg, &workingImg, 1);
        } else {
            workingImg = himg;
        }
        
        HTuple channels;
        CountChannels(workingImg, &channels);
        
        // DEBUG: Show image info
        QMessageBox::information(this, "Debug", QString("Obj Count: %1, Channels: %2").arg(count.I()).arg(channels[0].I()));

        HImage gray;
        if (channels[0].I() == 1) {
            gray = workingImg;
        } else if (channels[0].I() == 3) {
            Rgb1ToGray(workingImg, &gray);
        } else if (channels[0].I() == 4) {
            // RGBA: Decompose and convert RGB to Gray, ignoring Alpha
            HImage r, g, b, a;
            Decompose4(workingImg, &r, &g, &b, &a);
            HImage rgb;
            Compose3(r, g, b, &rgb);
            Rgb1ToGray(rgb, &gray);
        } else {
            // Fallback for other channel counts: just use the first channel
            AccessChannel(workingImg, &gray, 1);
        }
        
        HImage smooth;
        MedianImage(gray, &smooth, "circle", (std::max)(1, blurK), "mirrored");
        
        // --- Morphology & Region Analysis to isolate target and get outer contour ---
        
        // 1. Segmentation (Auto Thresholding)
        // Assuming backlighting/dark object on light background based on typical measurement scenarios
        HRegion region;
        HTuple usedThreshold;
        // "max_separability" is robust for bimodal histograms. "dark" selects the darker foreground.
        BinaryThreshold(smooth, &region, "max_separability", "dark", &usedThreshold);
        
        // 2. Morphology cleanup
        HRegion opened;
        // Remove small noise points
        OpeningCircle(region, &opened, 3.5);
        
        HRegion filled;
        // Fill internal holes to ensure we get a solid object
        FillUp(opened, &filled);
        
        HRegion connected;
        Connection(filled, &connected);
        
        // 3. Select the largest region (Target Object)
        HRegion selected;
        HTuple area, row, col;
        AreaCenter(connected, &area, &row, &col);
        if (area.Length() > 0) {
            HTuple maxIndex;
            TupleSortIndex(area, &maxIndex);
            // Select the region with max area
            SelectObj(connected, &selected, maxIndex[maxIndex.Length()-1].I() + 1);
        } else {
            selected = filled; // Fallback
        }
        
        // 4. Extract ROI for Edge Detection (Focus on the boundary)
        HRegion border;
        // Get the geometric boundary of the region
        Boundary(selected, &border, "inner");
        
        HRegion dilatedBorder;
        // Expand the boundary slightly to create a search band for Canny
        DilationCircle(border, &dilatedBorder, 5.0);
        
        HImage reducedImg;
        ReduceDomain(smooth, dilatedBorder, &reducedImg);
        
        // 5. Sub-pixel Edge Detection on the masked boundary
        HObject edges;
        EdgesSubPix(reducedImg, &edges, HTuple("canny"), HTuple(1.0), HTuple(low), HTuple(high));
        
        // Filter out very short edges (noise)
        HObject selectedEdges;
        SelectShapeXld(edges, &selectedEdges, "contlength", "and", 20, 99999);

        HTuple w, h;
        GetImageSize(gray, &w, &h);
        
        HImage base;
        GenImageConst(&base, HTuple("byte"), w, h);
        
        HImage painted;
        PaintXld(selectedEdges, base, &painted, HTuple(255));
        
        std::string output = "temp_result_halcon.png";
        WriteImage(painted, HTuple("png"), HTuple(0), HTuple(output.c_str()));
        
        displayImage(QString::fromStdString(output));
        
    } catch (HalconCpp::HException &except) {
        QMessageBox::critical(this, "Halcon 错误", QString("Halcon 异常: %1").arg(except.ErrorMessage().Text()));
    } catch (...) {
        QMessageBox::critical(this, "错误", "Halcon 处理中发生未知错误。");
    }
}
