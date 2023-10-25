#include <string>

using namespace std;
class FastCLARANSOutput {
public:
    double cost;
    int medoids_length;
    int* medoids;
    int results_length;
    int* results;
};

class SSIMFastCLARANSOutput {
public:
    double cost;
    
};

extern "C" {
    FastCLARANSOutput fast_clarans(double* dist, int n, int k, int numlocal=2, double maxneighbor=0.025, int seed = 123456789);
}

extern "C" {
    int ssim_fast_clarans(std::string sourceDir,int k, int numlocal=2, double maxneighbor=0.025, int seed = 123456789);
}