// This is cython file to generate python wrappper for C++ implementation of FastCLARANS

from libc.stdlib cimport free

cdef extern from "stdlib.h":
    void* malloc(size_t size)

cdef extern from "python_wrapper.hpp":
    cdef cppclass FastCLARANSOutput:
        double cost;
        int medoids_length;
        int* medoids;
        int results_length;
        int* results;

cdef extern from "python_wrapper.hpp":
    FastCLARANSOutput fast_clarans(double* dist, int n, int k, int numlocal, double maxneighbor, int seed);

def py_fast_clarans(dist,n,k,numlocal,maxneighbor,seed):
    cdef int length = len(dist)
    cdef double* dist_matrix = <double*>malloc(length * sizeof(double))
    
    if dist_matrix is NULL:
        raise MemoryError("Failed to allocate memory for data.")
    try:
        for i in range(length):
            dist_matrix[i] = dist[i]
    except:
        free(dist_matrix)
        raise MemoryError("Failed to allocate memory for data.")
    
    cdef FastCLARANSOutput fast_clarans_output = fast_clarans(dist_matrix, n, k, numlocal, maxneighbor, seed)

    cost = fast_clarans_output.cost
    medoids = [fast_clarans_output.medoids[i] for i in range(fast_clarans_output.medoids_length)]
    results = [fast_clarans_output.results[i] for i in range(fast_clarans_output.results_length)]

    free(fast_clarans_output.medoids)
    free(fast_clarans_output.results)
    free(dist_matrix)

    return cost, medoids, results
