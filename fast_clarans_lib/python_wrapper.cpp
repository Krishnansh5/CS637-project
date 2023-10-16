#include <vector>
#include <utility>
#include "pam.h"
#include "python_wrapper.hpp"
#include <iostream>

pair<int*,int> vectorToArray(const std::vector<int>& input_vector) {
    size_t length = input_vector.size();
    int* array = new int[length];
    
    for (size_t i = 0; i < length; ++i) {
        array[i] = input_vector[i];
    }

    return pair<int*,int>(array,length);
}

FastCLARANSOutput fast_clarans(double* dist, int n, int k, int numlocal, double maxneighbor, int seed) {
    std::cout<<"flag_0";
    std::vector<double> dist_matrix(dist, dist + n*(n+1)/2);
    // std::cout<<n<<endl;
    // for(auto v : dist_matrix){
    //     std::cout<<v<<" ";
    // }
    
    std::cout<<"flag_1";
    RDistMatrix dm(n, dist_matrix);

    std::cout<<"flag_2";
    FastCLARANS clarans(n, &dm, k, numlocal, maxneighbor, seed);

    std::cout<<"flag_3";
    FastCLARANSOutput ret;

    std::cout<<"flag_4";
    ret.cost = clarans.run();

    std::cout<<"flag_5";
    pair<int*,int> medoids = vectorToArray(clarans.getMedoids());
    ret.medoids_length = medoids.second;
    ret.medoids = medoids.first;

    pair<int*,int> results = vectorToArray(clarans.getResults());
    ret.results_length = results.second;
    ret.results = results.first;

    std::cout<<"flag_6";
    return ret;
}