#include "python_wrapper.hpp"
#include "pam.h"
#include "ssim.hpp"
#include "carla_data.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <ctime>
using namespace std;

void convertTensorToImg(torch::Tensor imageTensor){
    cv::Mat imageMat(imageTensor.sizes()[2], imageTensor.sizes()[3], CV_8UC4, imageTensor.data_ptr());
    cv::Mat rgbaImage = cv::Mat(imageMat.size(), CV_8UC4);
    cv::imshow("Resulting Image", rgbaImage);
    cv::waitKey(0);
}

void runSSIMFastClarans(string sourceDir,int k){
    ssim_fast_clarans(sourceDir, k, 2, 0.025, 123456789);
}

int main(int argc, char* argv[]){
    if (argc != 3) {
        std::cerr << "usage: ./fast_clarans <path-to-training-data-dir> <num-medoids>\n";
        return -1;
    }
    string sourceDir = argv[1];
    int k = atoi(argv[2]);

    clock_t start = clock();
    runSSIMFastClarans(sourceDir,k);

    clock_t end = clock();

    double duration = (static_cast<double>(end - start) / CLOCKS_PER_SEC); // in seconds
    cout << "Time taken: " << duration << " seconds" << endl;
};