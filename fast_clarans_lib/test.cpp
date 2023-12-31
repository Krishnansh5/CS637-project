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

void testSSIMFastClarans(string sourceDir,int k){
    // string sourceDir="/home/krish/CS637-project/interpretable_ood_detection/carla_experiments/training_data/setting_1/test_0";
    ssim_fast_clarans(sourceDir, k, 2, 0.025, 123456789);
}

int main(int argc, char* argv[]){
    // if (argc != 3) {
    //     std::cerr << "usage: ./fast_clarans <path-to-training-data-dir> <num-medoids>\n";
    //     return -1;
    // }
    // string sourceDir = argv[1];
    // int k = atoi(argv[2]);

    string sourceDir="/home/krish/CS637-project/interpretable_ood_detection/carla_experiments/training_data/setting_1/0";
    CarlaData carla_data(sourceDir);
    auto X=carla_data.imageTensorData[0];
    auto Y=carla_data.imageTensorData[1];
    int c = 2;
    auto X1 = c*X;
    auto Y1 = c*Y;
    auto z = ssim(X,Y);
    cout<<z<<endl;
    z = ssim(X1, Y1);
    cout<<z<<endl;

    // cout<<medoids.imageTensorData[0].sizes()<<endl;
    // cout<<medoids.imageTensorData[1].sizes()<<endl;
    // convertTensorToImg(medoids.imageTensorData[1]);
    // medoids.printAllPaths();
    // auto z = medoids.getSSIMDistance(0,1);

    // auto z=medoids.imageTensorData[0];
    // auto z = torch::rand({12, 12}, torch::TensorOptions(torch::kCPU).dtype(at::kFloat));
    // vector<float> tensorData(z.data_ptr<float>(), z.data_ptr<float>() + z.numel());
    // for (const auto& v : tensorData) {
    //     cout<<v<< " ";
    // }

    // torch::manual_seed(0);
    // std::string device = "cpu";
    // if(torch::cuda::is_available()) device = "cuda";
    // torch::Tensor X = torch::rand({1, 3, 32, 32}).to(device);
    // torch::Tensor Y = torch::rand({1, 3, 32, 32}).to(device);
    // auto z = ssim(X, Y);
    
    // cout<<z;


    // clock_t start = clock();
    // testSSIMFastClarans(sourceDir,k);

    // clock_t end = clock();

    // double duration = (static_cast<double>(end - start) / CLOCKS_PER_SEC); // in seconds
    // std::cout << "Time taken: " << duration << " seconds" << std::endl;
};