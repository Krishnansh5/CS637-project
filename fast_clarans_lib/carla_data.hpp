#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <vector>
#include <string>

using namespace std;

#ifndef MEDOIDS_H
#define MEDOIDS_H

class CarlaData {
    private:
        string source_dir;
        double** ssim_dist;
    public:
        int num_obs;
        std::vector<torch::Tensor> imageTensorData;
        std::vector<std::string> imagePaths;

        CarlaData(string source_dir);
        int laodAllImages(std::string sourceDir);
        torch::Tensor convertImageToTensor(cv::Mat& image);
        void printAllPaths();
        double getSSIMDistance(int i,int j);
    
        ~CarlaData(){
            for(int i=0;i<num_obs;i++){
                delete[] ssim_dist[i];
            }
            delete[] ssim_dist;
        }
};

#endif
