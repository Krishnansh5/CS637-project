#include <vector>
#include <utility>
#include "pam.h"
#include "python_wrapper.hpp"
#include "ssim.hpp"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

pair<int*,int> vectorToArray(const std::vector<int>& input_vector) {
    size_t length = input_vector.size();
    int* array = new int[length];
    
    for (size_t i = 0; i < length; ++i) {
        array[i] = input_vector[i];
    }

    return pair<int*,int>(array,length);
}

FastCLARANSOutput fast_clarans(double* dist, int n, int k, int numlocal, double maxneighbor, int seed) {
    std::vector<double> dist_matrix(dist, dist + n*(n+1)/2);

    RDistMatrix dm(n, dist_matrix);

    FastCLARANS clarans(n, &dm, k, numlocal, maxneighbor, seed);

    FastCLARANSOutput ret;
    ret.cost = clarans.run();

    pair<int*,int> medoids = vectorToArray(clarans.getMedoids());
    ret.medoids_length = medoids.second;
    ret.medoids = medoids.first;

    pair<int*,int> results = vectorToArray(clarans.getResults());
    ret.results_length = results.second;
    ret.results = results.first;

    return ret;
}