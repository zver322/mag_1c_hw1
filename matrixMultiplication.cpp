#include <iostream>
#include <random>
#include <iomanip>
#include <omp.h>
// #include <cblas.h>
#include <mkl.h>

struct Result {
    double serialCalculationTime;
    double parallelCalculationTime;
    double cblasCalculationTime;
    double mklCalculationTime;
};

template <typename T>
void fillRandom(T* matrix, size_t matrixSize, T xmin, T xmax, size_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(xmin, xmax);
    for (size_t i = 0; i != matrixSize; ++i) matrix[i] = dist(rng);
}

template <typename T>
void parallelMatrixMultiplication(const size_t M, const size_t N, const size_t K, const T* A, const T* B, T* C) {
    #pragma omp parallel for
    for (size_t i = 0; i != M * K; ++i) C[i] = 0;

    #pragma omp parallel for collapse(2)
    for (size_t m = 0; m != M; ++m) {
        for (size_t k = 0; k != K; ++k) {
            for (size_t n = 0; n != N; ++n) {
                C[m + k * M] += A[m + n * M] * B[n + k * N]; 
            }
        }
    }
}

template <typename T>
void serialMatrixMultiplication(const size_t M, const size_t N, const size_t K, const T* A, const T* B, T* C) {
    for (size_t i = 0; i != M * K; ++i) C[i] = 0;

    for (size_t m = 0; m != M; ++m) {
        for (size_t k = 0; k != K; ++k) {
            for (size_t n = 0; n != N; ++n) {
                C[m + k * M] += A[m + n * M] * B[n + k * N]; 
            }
        }
    }
}

template <typename T>
void printMatrix(const T* matrix, const size_t N) {
    for (size_t i = 0; i != N; ++i) {
        for (size_t j = 0; j != N; ++j) {
            std::cout << matrix[i + j * N] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
double sumMatrixElements(const T* martix, const size_t N) {
    double sum = 0.0;
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            sum += martix[i + j * N];
    return sum;
}

int main() {
    size_t N;
    std::cin >> N;
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    fillRandom(A, N * N, -1.0, 1.0, 9876);
    fillRandom(B, N * N, -1.0, 1.0, 9877);
    Result result;

  
    double t0 = omp_get_wtime();
    serialMatrixMultiplication(N, N, N, A, B, C);
    double t1 = omp_get_wtime();
    result.serialCalculationTime = t1 - t0;
    std::cout << "Serial Matrix Calculation. First element: " << C[0] << std::endl;
    // << sumMatrixElements(C, N) << std::endl;
    // printMatrix(C, N);

    t0 = omp_get_wtime();
    parallelMatrixMultiplication(N, N, N, A, B, C);
    t1 = omp_get_wtime();
    result.parallelCalculationTime = t1 - t0;
    std::cout << "Parallel Matrix Calculation. First element: " << C[0] << std::endl; 
    // << sumMatrixElements(C, N) << std::endl;
    // printMatrix(C, N);

    //t0 = omp_get_wtime();
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0, C, N);
    //t1 = omp_get_wtime();
    //result.cblasCalculationTime = t1 - t0;
    //std::cout << "OpenBlass Matrix Calculation: " << C[0] << std::endl; 
    // << sumMatrixElements(C, N) << std::endl;
    // printMatrix(C, N);

    t0 = omp_get_wtime();
    cblass_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0, C, N);
    t1 = omp_get_wtime();
    result.mklCalculationTime = t1 - t0;
    std::cout << "MKL Matrix Calculation: " << C[0] << std::endl;
    // << sumMatrixElements(C, N) << std::endl;
    // printMatrix(C, N);


    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Serial total time = " << result.serialCalculationTime << std::endl
              << "Parallel calculation time = " << result.parallelCalculationTime << std::endl
              //<< "Cblas calculation time = " << result.cblasCalculationTime << std::endl;
              << "MKL calculation time = " << result.mklCalculationTime << std::endl;
  
    delete[] A;
    delete[] B;
    delete[] C;
}
