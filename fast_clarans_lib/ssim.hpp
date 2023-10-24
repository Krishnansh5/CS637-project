#pragma once

#include <torch/torch.h>

torch::Tensor fspecial_gauss(int size, float sigma);

torch::Tensor gaussian_filter_2d(torch::Tensor inp, torch::Tensor win);

std::pair<torch::Tensor, torch::Tensor> _ssim(
    torch::Tensor X,
    torch::Tensor Y,
    float data_range,
    torch::Tensor win,
    bool size_average=true,
    std::pair<float, float> K={0.01, 0.03}
);

torch::Tensor ssim(
    torch::Tensor X,
    torch::Tensor Y,
    float data_range=255.0,
    bool size_average=true,
    int win_size = 11,
    float win_sigma = 1.5,
    std::pair<float, float> K = {0.01, 0.03},
    bool nonnegative_ssim = false
);
