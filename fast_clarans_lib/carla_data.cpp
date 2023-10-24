#include "carla_data.hpp"
#include <string>
#include "ssim.hpp"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

using namespace std;

CarlaData::CarlaData(int num_obs,string source_dir) : num_obs(num_obs),source_dir(source_dir) {
    laodAllImages(source_dir);
    ssim_dist = new double*[num_obs];
    for(int i=0;i<num_obs;i++){
        ssim_dist[i] = new double[num_obs];
    }
    for(int i=0;i<num_obs;i++){
        for(int j=0;j<num_obs;j++){
            ssim_dist[i][j] = -1;
        }
    }
}

void CarlaData::laodAllImages(string sourceDir) {
    for (const auto& entry : std::filesystem::directory_iterator(sourceDir)) {
        if(entry.is_directory()){
            laodAllImages(entry.path().string());
        }
        if(entry.path().extension() != ".png"){
            continue;
        }
        cv::Mat image = cv::imread(entry.path(),cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << entry.path() << std::endl;
            return;
        }
        torch::Tensor tensor = convertImageToTensor(image);
        imageTensorData.push_back(tensor);
        imagePaths.push_back(entry.path());
    }
}

torch::Tensor CarlaData::convertImageToTensor(cv::Mat& image){
    std::string device = "cpu";
    if(torch::cuda::is_available()) device = "cuda";

    cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    cv::resize(image, image, cv::Size(128, 128));

    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F, 1.0 / 255);
    auto size = imageFloat.size();

    torch::Tensor imageTensor = torch::from_blob(
        imageFloat.data,
        {1, 4, size.height, size.width} // batch_size(1), n_channels(4 - BGRA), height(128), width(128)
    );

    return imageTensor.to(device).clone();
}

void CarlaData::printAllPaths(){
    for(auto v : imagePaths){
        cout<<v<<"\n";
    }
}

double CarlaData::getSSIMDistance(int i,int j){
    if(ssim_dist[i][j] != -1){
        return ssim_dist[i][j];
    }
    ssim_dist[i][j] = ssim(imageTensorData[i],imageTensorData[j]).item<double>();
    return ssim_dist[i][j];
}