/// @file dense_algebra.cpp
/// @brief numpy's dense operations on the OpenBLAS numpy ships (scipy-openblas64 0.3.29), called as numpy calls them.

#include "dense_algebra.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

// The exported names of scipy-openblas64 (its lapack.h macros leave out the "scipy_" prefix, so they are declared here).
extern "C" {
void scipy_cblas_dgemm64_(int order, int trans_a, int trans_b, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double* a,
                          std::int64_t lda, const double* b, std::int64_t ldb, double beta, double* c, std::int64_t ldc);
void scipy_cblas_dsyrk64_(int order, int uplo, int trans, std::int64_t n, std::int64_t k, double alpha, const double* a, std::int64_t lda, double beta,
                          double* c, std::int64_t ldc);
double scipy_cblas_ddot64_(std::int64_t n, const double* x, std::int64_t incx, const double* y, std::int64_t incy);
void scipy_cblas_dgemv64_(int order, int trans, std::int64_t m, std::int64_t n, double alpha, const double* a, std::int64_t lda, const double* x,
                          std::int64_t incx, double beta, double* y, std::int64_t incy);
void scipy_dgesv_64_(const std::int64_t* n, const std::int64_t* nrhs, double* a, const std::int64_t* lda, std::int64_t* ipiv, double* b,
                     const std::int64_t* ldb, std::int64_t* info);
void scipy_dgesdd_64_(const char* jobz, const std::int64_t* m, const std::int64_t* n, double* a, const std::int64_t* lda, double* s, double* u,
                      const std::int64_t* ldu, double* vt, const std::int64_t* ldvt, double* work, const std::int64_t* lwork, std::int64_t* iwork,
                      std::int64_t* info, std::size_t jobz_length);
void scipy_openblas_set_num_threads64_(int num_threads);
}

namespace fluidx3d::experimental::wall_builder {

namespace {
constexpr int ROW_MAJOR{ 101 };
constexpr int NO_TRANS{ 111 };
constexpr int TRANS{ 112 };
constexpr int UPPER{ 121 };

/// @brief A column-major copy of a row-major matrix (numpy linearizes into Fortran order before LAPACK).
std::vector<double> column_major(const Matrix& a)
{
    std::vector<double> out(a.data.size());
    for (int r{ 0 }; r < a.rows; ++r) {
        for (int c{ 0 }; c < a.cols; ++c) {
            out[static_cast<std::size_t>(c) * static_cast<std::size_t>(a.rows) + static_cast<std::size_t>(r)] = a.at(r, c);
        }
    }
    return out;
}
} // namespace

Matrix DenseAlgebra::transpose_times(const Matrix& a, const Matrix& b)
{
    if (a.rows != b.rows) {
        throw std::invalid_argument("transpose_times: " + std::to_string(a.rows) + " and " + std::to_string(b.rows) + " rows");
    }
    Matrix out{ a.cols, b.cols, std::vector<double>(static_cast<std::size_t>(a.cols) * static_cast<std::size_t>(b.cols), 0.0) };
    scipy_cblas_dgemm64_(ROW_MAJOR, TRANS, NO_TRANS, a.cols, b.cols, a.rows, 1.0, a.data.data(), a.cols, b.data.data(), b.cols, 0.0, out.data.data(),
                         b.cols);
    return out;
}

Matrix DenseAlgebra::gram(const Matrix& a)
{
    const int n{ a.cols };
    Matrix out{ n, n, std::vector<double>(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0) };
    scipy_cblas_dsyrk64_(ROW_MAJOR, UPPER, TRANS, n, a.rows, 1.0, a.data.data(), n, 0.0, out.data.data(), n);
    for (int i{ 0 }; i < n; ++i) {
        for (int j{ i + 1 }; j < n; ++j) {
            out.data[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)]
                = out.data[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
        }
    }
    return out;
}

Matrix DenseAlgebra::times(const Matrix& a, const Matrix& b)
{
    if (a.cols != b.rows) {
        throw std::invalid_argument("times: " + std::to_string(a.cols) + " columns and " + std::to_string(b.rows) + " rows");
    }
    Matrix out{ a.rows, b.cols, std::vector<double>(static_cast<std::size_t>(a.rows) * static_cast<std::size_t>(b.cols), 0.0) };
    scipy_cblas_dgemm64_(ROW_MAJOR, NO_TRANS, NO_TRANS, a.rows, b.cols, a.cols, 1.0, a.data.data(), a.cols, b.data.data(), b.cols, 0.0, out.data.data(),
                         b.cols);
    return out;
}

Matrix DenseAlgebra::times_transposed(const Matrix& a, const Matrix& b)
{
    if (a.cols != b.cols) {
        throw std::invalid_argument("times_transposed: " + std::to_string(a.cols) + " and " + std::to_string(b.cols) + " columns");
    }
    Matrix out{ a.rows, b.rows, std::vector<double>(static_cast<std::size_t>(a.rows) * static_cast<std::size_t>(b.rows), 0.0) };
    scipy_cblas_dgemm64_(ROW_MAJOR, NO_TRANS, TRANS, a.rows, b.rows, a.cols, 1.0, a.data.data(), a.cols, b.data.data(), b.cols, 0.0, out.data.data(),
                         b.rows);
    return out;
}

double DenseAlgebra::dot(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot: vectors of " + std::to_string(x.size()) + " and " + std::to_string(y.size()));
    }
    return scipy_cblas_ddot64_(static_cast<std::int64_t>(x.size()), x.data(), 1, y.data(), 1);
}

std::vector<double> DenseAlgebra::times_vector(const Matrix& a, const std::vector<double>& v)
{
    if (static_cast<std::size_t>(a.cols) != v.size()) {
        throw std::invalid_argument("times_vector: " + std::to_string(a.cols) + " columns and a vector of " + std::to_string(v.size()));
    }
    std::vector<double> out(static_cast<std::size_t>(a.rows), 0.0);
    scipy_cblas_dgemv64_(ROW_MAJOR, NO_TRANS, a.rows, a.cols, 1.0, a.data.data(), a.cols, v.data(), 1, 0.0, out.data(), 1);
    return out;
}

std::vector<double> DenseAlgebra::vector_times(const std::vector<double>& v, const Matrix& a)
{
    if (static_cast<std::size_t>(a.rows) != v.size()) {
        throw std::invalid_argument("vector_times: a vector of " + std::to_string(v.size()) + " and " + std::to_string(a.rows) + " rows");
    }
    std::vector<double> out(static_cast<std::size_t>(a.cols), 0.0);
    scipy_cblas_dgemv64_(ROW_MAJOR, TRANS, a.rows, a.cols, 1.0, a.data.data(), a.cols, v.data(), 1, 0.0, out.data(), 1);
    return out;
}

std::vector<double> DenseAlgebra::solve(const Matrix& a, const std::vector<double>& b)
{
    const Matrix rhs{ static_cast<int>(b.size()), 1, b };
    const Matrix x{ solve(a, rhs) };
    return x.data;
}

Matrix DenseAlgebra::solve(const Matrix& a, const Matrix& b)
{
    if (a.rows != a.cols || b.rows != a.rows) {
        throw std::invalid_argument("solve: a " + std::to_string(a.rows) + " x " + std::to_string(a.cols) + " system with " + std::to_string(b.rows) + " rows");
    }
    std::vector<double> lu{ column_major(a) };
    std::vector<double> x{ column_major(b) };
    const std::int64_t n{ a.rows };
    const std::int64_t nrhs{ b.cols };
    std::vector<std::int64_t> pivots(static_cast<std::size_t>(n), 0);
    std::int64_t info{ 0 };
    scipy_dgesv_64_(&n, &nrhs, lu.data(), &n, pivots.data(), x.data(), &n, &info);
    if (info != 0) {
        throw std::domain_error("Singular matrix (dgesv info " + std::to_string(info) + ")");
    }
    Matrix out{ b.rows, b.cols, std::vector<double>(b.data.size(), 0.0) };
    for (int r{ 0 }; r < b.rows; ++r) {
        for (int c{ 0 }; c < b.cols; ++c) {
            out.data[static_cast<std::size_t>(r) * static_cast<std::size_t>(b.cols) + static_cast<std::size_t>(c)]
                = x[static_cast<std::size_t>(c) * static_cast<std::size_t>(b.rows) + static_cast<std::size_t>(r)];
        }
    }
    return out;
}

std::vector<double> DenseAlgebra::singular_values(const Matrix& a)
{
    std::vector<double> work_matrix{ column_major(a) };
    const std::int64_t m{ a.rows };
    const std::int64_t n{ a.cols };
    const std::int64_t k{ std::min(m, n) };
    const std::int64_t one{ 1 };
    const char jobz{ 'N' };
    std::vector<double> s(static_cast<std::size_t>(k), 0.0);
    std::vector<std::int64_t> iwork(static_cast<std::size_t>(8 * k), 0);
    double unused{ 0.0 };
    double query{ 0.0 };
    const std::int64_t query_size{ -1 };
    std::int64_t info{ 0 };
    // the workspace query first, as numpy's umath_linalg does
    scipy_dgesdd_64_(&jobz, &m, &n, work_matrix.data(), &m, s.data(), &unused, &one, &unused, &one, &query, &query_size, iwork.data(), &info, 1);
    const auto lwork{ static_cast<std::int64_t>(query) };
    std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
    scipy_dgesdd_64_(&jobz, &m, &n, work_matrix.data(), &m, s.data(), &unused, &one, &unused, &one, work.data(), &lwork, iwork.data(), &info, 1);
    if (info != 0) {
        throw std::domain_error("SVD did not converge (dgesdd info " + std::to_string(info) + ")");
    }
    return s;
}

int DenseAlgebra::rank(const Matrix& a)
{
    const std::vector<double> s{ singular_values(a) };
    const double largest{ *std::max_element(s.begin(), s.end()) };
    const double rtol{ static_cast<double>(std::max(a.rows, a.cols)) * std::numeric_limits<double>::epsilon() };
    const double tol{ largest * rtol };
    int count{ 0 };
    for (const double value : s) {
        if (value > tol) {
            ++count;
        }
    }
    return count;
}

double DenseAlgebra::condition(const Matrix& a)
{
    const std::vector<double> s{ singular_values(a) };
    return s.front() / s.back();
}

void DenseAlgebra::single_threaded()
{
    scipy_openblas_set_num_threads64_(1);
}

} // namespace fluidx3d::experimental::wall_builder
