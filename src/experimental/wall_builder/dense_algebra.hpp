#pragma once
/// @file dense_algebra.hpp
/// @brief The small dense algebra of terrain_container.py, done with the OpenBLAS that numpy uses and called the way numpy
/// calls it, so every result is the same binary64 value as in Python.
///
/// numpy 2.2.6 ships scipy-openblas64 0.3.29 (ILP64, symbols prefixed "scipy_" and suffixed "64_"). Its kernels sum in
/// their own order and use FMA, so plain loops differ (equal in 5 of 20000 normal matrices); the same library called the
/// same way is equal in 20000 of 20000 (dgemm, dgesv, dgemv; 2026-11-12). Matrices here are row-major (C order), as numpy's.

#include <cstdint>
#include <vector>

namespace fluidx3d::experimental::wall_builder {

/// @brief A row-major matrix of binary64 values.
struct Matrix {
    int rows{};                  ///< Rows.
    int cols{};                  ///< Columns.
    std::vector<double> data{};  ///< rows * cols values, row-major.

    /// @brief The value at (r, c), 0-based.
    [[nodiscard]] double at(const int r, const int c) const { return data[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c)]; }
};

/// @brief numpy's dense operations on OpenBLAS: each method names the numpy expression it reproduces.
class DenseAlgebra {
public:
    /// @brief a.T @ b for C-order a (k x m) and b (k x n): cblas_dgemm(RowMajor, Trans, NoTrans).
    [[nodiscard]] static Matrix transpose_times(const Matrix& a, const Matrix& b);

    /// @brief a.T @ a for C-order a (k x n): numpy calls cblas_dsyrk(RowMajor, Upper, Trans) and mirrors the upper triangle.
    [[nodiscard]] static Matrix gram(const Matrix& a);

    /// @brief a @ b for C-order a (m x k) and b (k x n): cblas_dgemm(RowMajor, NoTrans, NoTrans).
    [[nodiscard]] static Matrix times(const Matrix& a, const Matrix& b);

    /// @brief a @ b.T for C-order a (m x k) and b (n x k): cblas_dgemm(RowMajor, NoTrans, Trans).
    [[nodiscard]] static Matrix times_transposed(const Matrix& a, const Matrix& b);

    /// @brief np.dot(x, y) of two vectors: cblas_ddot.
    [[nodiscard]] static double dot(const std::vector<double>& x, const std::vector<double>& y);

    /// @brief a @ v for C-order a (m x n) and a vector v (n): cblas_dgemv(RowMajor, NoTrans).
    [[nodiscard]] static std::vector<double> times_vector(const Matrix& a, const std::vector<double>& v);

    /// @brief v @ a for a vector v (m) and C-order a (m x n): cblas_dgemv(RowMajor, Trans).
    [[nodiscard]] static std::vector<double> vector_times(const std::vector<double>& v, const Matrix& a);

    /// @brief np.linalg.solve(a, b) for a square a and one right-hand side: dgesv on a column-major copy.
    /// @throws std::domain_error when a is singular (numpy raises LinAlgError)
    [[nodiscard]] static std::vector<double> solve(const Matrix& a, const std::vector<double>& b);

    /// @brief np.linalg.solve(a, b) for a square a and a matrix b of right-hand sides (C order): dgesv, column-major copies.
    /// @throws std::domain_error when a is singular
    [[nodiscard]] static Matrix solve(const Matrix& a, const Matrix& b);

    /// @brief np.linalg.svd(a, compute_uv=False): dgesdd (jobz 'N') on a column-major copy, descending.
    /// @throws std::domain_error when the SVD does not converge
    [[nodiscard]] static std::vector<double> singular_values(const Matrix& a);

    /// @brief np.linalg.matrix_rank(a): the singular values above max(S) * max(rows, cols) * eps.
    [[nodiscard]] static int rank(const Matrix& a);

    /// @brief np.linalg.cond(a) (2-norm): S[0] / S[-1].
    [[nodiscard]] static double condition(const Matrix& a);

    /// @brief Makes OpenBLAS run each call on the calling thread (the builder runs many calls in parallel itself).
    static void single_threaded();
};

} // namespace fluidx3d::experimental::wall_builder
