class FastCLARANSOutput {
public:
    double cost;
    int medoids_length;
    int* medoids;
    int results_length;
    int* results;
};

extern "C" {
    FastCLARANSOutput fast_clarans(double* dist, int n, int k, int numlocal=2, double maxneighbor=0.025, int seed = 123456789);
}