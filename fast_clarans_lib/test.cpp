#include "python_wrapper.hpp"
#include "pam.h"
#include "ssim.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
using namespace std;

void convertTensorToImg(torch::Tensor imageTensor){
    cv::Mat imageMat(imageTensor.sizes()[2], imageTensor.sizes()[3], CV_8UC4, imageTensor.data_ptr());
    cv::Mat rgbaImage = cv::Mat(imageMat.size(), CV_8UC4);
    cv::imshow("Resulting Image", rgbaImage);
    cv::waitKey(0);
}

int main(){
    string sourceDir="/home/krish/CS637-project/interpretable_ood_detection/carla_experiments/training_data/setting_1/0";
    Medoids medoids;
    medoids.laodAllImages(sourceDir);
    // cout<<medoids.imageTensorData[0].sizes()<<endl;
    // cout<<medoids.imageTensorData[1].sizes()<<endl;
    // convertTensorToImg(medoids.imageTensorData[1]);
    // medoids.printAllPaths();
    auto z = medoids.getSSIMDistance(0,1);

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
    
    cout<<z;
};