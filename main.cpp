#include <QApplication>
#include "MainWindow.h"

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <halconcpp/HalconCpp.h>
#include <string>
#include <iostream>

int main(int argc, char** argv) {
    // If arguments are provided, try to run in CLI mode (legacy support)
    // Note: Qt takes over some arguments, so this simple check might need refinement if mixed.
    // For now, if we have specific CLI args, we could run CLI, otherwise GUI.
    // However, since we are integrating Qt, let's prioritize GUI if no specific "cli" flag or if args are insufficient.
    
    bool runGui = true;
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" || arg == "--help") {
            runGui = false;
            break;
        }
    }

    if (runGui) {
        QApplication app(argc, argv);
        MainWindow w;
        w.show();
        return app.exec();
    }

    // CLI Implementation (Legacy)
    std::string backend;
    std::string input;
    std::string output;
    int low = 70;
    int high = 150;
    int blurK = 5;
    bool show = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) { backend = argv[++i]; }
        else if (a == "--input" && i + 1 < argc) { input = argv[++i]; }
        else if (a == "--output" && i + 1 < argc) { output = argv[++i]; }
        else if (a == "--low" && i + 1 < argc) { low = std::stoi(argv[++i]); }
        else if (a == "--high" && i + 1 < argc) { high = std::stoi(argv[++i]); }
        else if (a == "--blur" && i + 1 < argc) { blurK = std::stoi(argv[++i]); if (blurK % 2 == 0) ++blurK; }
        else if (a == "--show") { show = true; }
    }
    
    if (backend.empty() || input.empty()) {
        std::cerr << "usage: OpenCVHalconCanny --backend opencv|halcon --input <img> --output <png> [--low N --high N --blur K --show]" << std::endl;
        return 1;
    }
    
    if (output.empty()) { output = "canny_preview.png"; }
    
    if (backend == "opencv") {
#ifndef HAVE_OPENCV
        std::cerr << "opencv backend unavailable" << std::endl; return 4;
#else
        cv::Mat src = cv::imread(input, cv::IMREAD_COLOR);
        if (src.empty()) { std::cerr << "read fail" << std::endl; return 2; }
        cv::Mat gray; cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        if (blurK > 1) cv::GaussianBlur(gray, gray, cv::Size(blurK, blurK), blurK * 0.2);
        cv::Mat edges; cv::Canny(gray, edges, low, high);
        cv::imwrite(output, edges);
        if (show) { cv::imshow("opencv_canny", edges); cv::waitKey(0); }
        return 0;
#endif
    } else if (backend == "halcon") {
        using namespace HalconCpp;
        try {
            HImage himg; ReadImage(&himg, HTuple(input.c_str()));
            HImage gray; Rgb1ToGray(himg, &gray);
            HImage smooth; MedianImage(gray, &smooth, "circle", (std::max)(1, blurK), "mirrored");
            HObject edges; EdgesSubPix(smooth, &edges, HTuple("canny"), HTuple(1.0), HTuple(low), HTuple(high));
            HTuple w, h; GetImageSize(gray, &w, &h);
            HImage base; GenImageConst(&base, HTuple("byte"), w, h);
            HImage painted; PaintXld(edges, base, &painted, HTuple(255));
            WriteImage(painted, HTuple("png"), HTuple(0), HTuple(output.c_str()));
            if (show) {
#ifdef HAVE_OPENCV
                cv::Mat view = cv::imread(output, cv::IMREAD_GRAYSCALE);
                if (!view.empty()) { cv::imshow("halcon_canny", view); cv::waitKey(0); }
#endif
            }
            return 0;
        } catch (HalconCpp::HException &except) {
            std::cerr << "Halcon error: " << except.ErrorMessage().Text() << std::endl;
            return 5;
        }
    } else {
        std::cerr << "backend invalid" << std::endl;
        return 3;
    }
}
