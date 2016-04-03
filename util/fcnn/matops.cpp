/*
 *  This file is a part of Fast Compressed Neural Networks.
 *
 *  Copyright (c) Grzegorz Klima 2008-2016
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

/** \file matops.cpp
 *  \brief Matrix operations.
 */




#include <mat.h>
#include <matops.h>
#include <utils.h>
#include <error.h>
#include <level1.h>
#include <iomanip>
#include <vector>
#include <cmath>


using namespace fcnn;
using namespace fcnn::internal;


// ==================================================================
// Unary minus
// ==================================================================
namespace {


template <typename T>
struct uminus {
};


template <typename T>
struct uminus<Matrix<T> > {
    static Matrix<T>
    eval(const Matrix<T> &m) {
        int r = m.rows(), c = m.cols(), s = m.size();
        if (!s) return m;
        Matrix<T> res(r, c);
        for (int i = 1; i <= s; ++i)
            res.elem(i) = -m.elem(i);
        return res;
    }
};


} /* namespace */



// Definition
// ==================================================================
template <typename T>
typename fcnn::internal::enable_unary<T>::type
fcnn::operator-(const T &m)
{
    return uminus<T>::eval(m);
}

// Instantiations
// ==================================================================
template typename fcnn::internal::enable_unary<Matrix<float> >::type
fcnn::operator-(const Matrix<float>&);
template typename fcnn::internal::enable_unary<Matrix<double> >::type
fcnn::operator-(const Matrix<double>&);




// ==================================================================
// Transposition
// ==================================================================
namespace {


template <typename T>
struct trans {
};


template <typename T>
struct trans<Matrix<T> > {
    static Matrix<T>
    eval(const Matrix<T> &m) {
        int r = m.rows(), c = m.cols(), s = m.size();
        Matrix<T> res(c, r);
        if (!s) return res;

        for (int i = 1; i <= r; ++i) {
            for (int j = 1; j <= c; ++j) {
                res.elem(j, i) = m.elem(i, j);
            }
        }

        return res;
    }
};


} /* namespace */



// Definition
// ==================================================================
template <typename T>
typename fcnn::internal::enable_unary<T>::type
fcnn::t(const T &m)
{
    return trans<T>::eval(m);
}

// Instantiations
// ==================================================================
template typename fcnn::internal::enable_unary<Matrix<float> >::type
fcnn::t(const Matrix<float>&);
template typename fcnn::internal::enable_unary<Matrix<double> >::type
fcnn::t(const Matrix<double>&);




// ==================================================================
// Reshape
// ==================================================================
namespace {


template <typename T>
struct reshap {
};


template <typename T>
struct reshap<Matrix<T> > {
    static Matrix<T>
    eval(const Matrix<T> &mat, int m, int n) {
        int r = mat.rows(), c = mat.cols(), s = mat.size();
        if (m * n != s) {
            message mes;
            mes << "nonconformant sizes in reshape; original size is "
                << r << 'x' << c << ", requested " << m << 'x' << n;
            error(mes);
        }
        Matrix<T> res(m, n, mat.ptr());
        return res;
    }
};


} /* namespace */


// Definition
// ==================================================================
template <typename T>
typename fcnn::internal::enable_unary<T>::type
fcnn::reshape(const T &mat, int m, int n)
{
    return reshap<T>::eval(mat, m, n);
}

// Instantiations
// ==================================================================
template typename fcnn::internal::enable_unary<Matrix<float> >::type
fcnn::reshape(const Matrix<float>&, int, int);
template typename fcnn::internal::enable_unary<Matrix<double> >::type
fcnn::reshape(const Matrix<double>&, int, int);




// ==================================================================
// +, -, mul, div (element by element ops)
// ==================================================================
namespace {

template <typename T>
struct op_add {
    static inline T eval(const T &a, const T &b) { return a + b; }
};

template <typename T>
struct op_sub {
    static inline T eval(const T &a, const T &b) { return a - b; }
};

template <typename T>
struct op_mul {
    static inline T eval(const T &a, const T &b) { return a * b; }
};

template <typename T>
struct op_div {
    static inline T eval(const T &a, const T &b) { return a / b; }
};




template <typename TA, typename TB, template <typename> class OP>
struct elbyel {
};

template <typename T, template <typename> class OP>
struct elbyel<Matrix<T>, Matrix<T>, OP> {
    static Matrix<T> eval(const Matrix<T> &A, const Matrix<T> &B) {
        int Ar = A.rows(), Ac = A.cols(), Br = B.rows(), Bc = B.cols(),
                 s = A.size();
        if ((Ar == 1) && (Ac == 1))
            return elbyel<T, Matrix<T>, OP>::eval(A.elem(1), B);
        if ((Br == 1) && (Bc == 1))
            return elbyel<Matrix<T>, T, OP>::eval(A, B.elem(1));
        if ((Ar != Br) || (Ac != Bc)) {
            message mes;
            mes << "nonconformant sizes in +,- or element by element "
                << "(mul, div) operation; 1st operand is "
                << Ar << 'x' << Ac << ", 2nd " << Br << 'x' << Bc;
            error(mes);
        }
        if (!s) return A;
        Matrix<T> res(Ar, Ac);
        for (int i = 1; i <= s; ++i)
            res.elem(i) = OP<T>::eval(A.elem(i), B.elem(i));
        return res;
    }
};

template <typename T, template <typename> class OP>
struct elbyel<T, Matrix<T>, OP> {
    static Matrix<T> eval(const T &A, const Matrix<T> &B) {
        int s = B.size();
        if (!s) return B;
        Matrix<T> res(B.rows(), B.cols());
        for (int i = 1; i <= s; ++i)
            res.elem(i) = OP<T>::eval(A, B.elem(i));
        return res;
    }
};

template <typename T, template <typename> class OP>
struct elbyel<Matrix<T>, T, OP> {
    static Matrix<T> eval(const Matrix<T> &A, const T &B) {
        int s = A.size();
        if (!s) return A;
        Matrix<T> res(A.rows(), A.cols());
        for (int i = 1; i <= s; ++i)
            res.elem(i) = OP<T>::eval(A.elem(i), B);
        return res;
    }
};

} /* namespace */



// Definitions
// ==================================================================
template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
fcnn::operator+(const TA &A, const TB &B)
{
    return elbyel<TA, TB, op_add>::eval(A, B);
}

template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
fcnn::operator-(const TA &A, const TB &B)
{
    return elbyel<TA, TB, op_sub>::eval(A, B);
}

template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
fcnn::mul(const TA &A, const TB &B)
{
    return elbyel<TA, TB, op_mul>::eval(A, B);
}

template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
fcnn::div(const TA &A, const TB &B)
{
    return elbyel<TA, TB, op_div>::eval(A, B);
}

// Instantiations
// ==================================================================
#define INSTANTIATE_BIN(WHAT) \
template \
typename fcnn::internal::enable_binary<Matrix<float>, Matrix<float> >::type \
fcnn::WHAT(const Matrix<float>&, const Matrix<float>&); \
template \
typename fcnn::internal::enable_binary<float, Matrix<float> >::type \
fcnn::WHAT(const float&, const Matrix<float>&); \
template \
typename fcnn::internal::enable_binary<Matrix<float>, float>::type \
fcnn::WHAT(const Matrix<float>&, const float&); \
template \
typename fcnn::internal::enable_binary<Matrix<double>, Matrix<double> >::type \
fcnn::WHAT(const Matrix<double>&, const Matrix<double>&); \
template \
typename fcnn::internal::enable_binary<double, Matrix<double> >::type \
fcnn::WHAT(const double&, const Matrix<double>&); \
template \
typename fcnn::internal::enable_binary<Matrix<double>, double>::type \
fcnn::WHAT(const Matrix<double>&, const double&);

INSTANTIATE_BIN(operator+)
INSTANTIATE_BIN(operator-)
INSTANTIATE_BIN(mul)
INSTANTIATE_BIN(div)






// ==================================================================
// * (matrix multiplication)
// ==================================================================
namespace {

template <typename TA, typename TB>
struct mat_mul {
};


template <typename T>
struct mat_mul<T, Matrix<T> > {
    static Matrix<T> eval(const T &A, const Matrix<T> &B) {
        int s = B.size();
        if (!s) return B;
        Matrix<T> res(B.rows(), B.cols());
        for (int i = 1; i <= s; ++i)
            res.elem(i) = A * B.elem(i);
        return res;
    }
};

template <typename T>
struct mat_mul<Matrix<T>, T> {
    static Matrix<T> eval(const Matrix<T> &A, const T &B) {
        int s = A.size();
        if (!s) return A;
        Matrix<T> res(A.rows(), A.cols());
        for (int i = 1; i <= s; ++i)
            res.elem(i) = A.elem(i) * B;
        return res;
    }
};


#if defined(HAVE_BLAS)
extern "C" {
float
F77_FUNC(sdot,SDOT)(const int *n,
                    const float *dx, const int *incx,
                    const float *dy, const int *incy);
double
F77_FUNC(ddot,DDOT)(const int *n,
                    const double *dx, const int *incx,
                    const double *dy, const int *incy);
void
F77_FUNC(sgemv,SGEMV)(char *trans, int *M, int *N, float *alpha,
                      const float* A, int *lda,
                      const float* x, int *incx,
                      float *beta, float* y, int *incy);
void
F77_FUNC(dgemv,DGEMV)(char *trans, int *M, int *N, double *alpha,
                      const double* A, int *lda,
                      const double* x, int *incx,
                      double *beta, double* y, int *incy);
void
F77_FUNC(sger,SGER)(int *M, int *N, float *alpha,
                    const float *x, int *incx, const float *y, int *incy,
                    float *A, int *lda);
void
F77_FUNC(dger,DGER)(int *M, int *N, double *alpha,
                    const double *x, int *incx, const double *y, int *incy,
                    double *A, int *lda);
void
F77_FUNC(sgemm,SGEMM)(char *transa, char *transb, int *M, int *N,
                      int *K, float *alpha, const float *A, int *lda,
                      const float *B, int *ldb, float *beta, float *C,
                      int *ldc);
void
F77_FUNC(dgemm,DGEMM)(char *transa, char *transb, int *M, int *N,
                      int *K, double *alpha, const double *A, int *lda,
                      const double *B, int *ldb, double *beta, double *C,
                      int *ldc);
} /* extern "C" */



inline
float
dot(int n, const float* x, int incx, const float* y, int incy)
{
    return F77_FUNC(sdot,SDOT)(&n, x, &incx, y, &incy);
}


inline
double
dot(int n, const double* x, int incx, const double* y, int incy)
{
    return F77_FUNC(ddot,DDOT)(&n, x, &incx, y, &incy);
}


inline
void
gemv(char trans, int M, int N, float alpha,
     const float* A, int lda,
     const float* x, int incx,
     float beta, float* y, int incy)
{
    F77_FUNC(sgemv,SGEMV)(&trans, &M, &N,
                          &alpha, A, &lda, x, &incx,
                          &beta, y, &incy);
}


inline
void
gemv(char trans, int M, int N, double alpha,
     const double* A, int lda,
     const double* x, int incx,
     double beta, double* y, int incy)
{
    F77_FUNC(dgemv,DGEMV)(&trans, &M, &N,
                          &alpha, A, &lda, x, &incx,
                          &beta, y, &incy);
}


inline
void
ger(int M, int N, float alpha,
    const float *x, int incx, const float *y, int incy,
    float *A, int lda)
{
    F77_FUNC(sger,SGER)(&M, &N, &alpha,
                        x, &incx, y, &incy,
                        A, &lda);
}


inline
void
ger(int M, int N, double alpha,
    const double *x, int incx, const double *y, int incy,
    double *A, int lda)
{
    F77_FUNC(dger,DGER)(&M, &N, &alpha,
                        x, &incx, y, &incy,
                        A, &lda);
}


inline
void
gemm(char transa, char transb, int M, int N,
     int K, float alpha, const float *A, int lda,
     const float *B, int ldb, float beta, float *C, int ldc)
{
    F77_FUNC(sgemm,SGEMM)(&transa, &transb, &M, &N, &K, &alpha, A, &lda,
                          B, &ldb, &beta, C, &ldc);
}


inline
void
gemm(char transa, char transb, int M, int N,
     int K, double alpha, const double *A, int lda,
     const double *B, int ldb, double beta, double *C, int ldc)
{
    F77_FUNC(dgemm,DGEMM)(&transa, &transb, &M, &N, &K, &alpha, A, &lda,
                          B, &ldb, &beta, C, &ldc);
}


#endif /* defined(HAVE_BLAS) */



template <typename T>
struct mat_mul<Matrix<T>, Matrix<T> > {
    static Matrix<T> eval(const Matrix<T> &A, const Matrix<T> &B) {
        int M = A.rows(), K = A.cols(), Br = B.rows(), N = B.cols();
        if (K != Br) {
            message mes;
            mes << "nonconformant sizes in matrix multiplication; "
                << "1st operand is " << M << 'x' << K
                << ", 2nd " << Br << 'x' << N;
            error(mes);
        }

        Matrix<T> res(M, N, (T)0.);
        if (!M || !K || !N) return res;
        const T *Aptr = A.ptr();
        const T *Bptr = B.ptr();
        T *Cptr = res.ptr();
#if defined(HAVE_BLAS)
        if ((M == 1) && (N == 1)) { // rowvector * vector
            res.elem(1) = dot(K, Aptr, 1, Bptr, 1);
        } else if (K == 1) { // vector * rowvector
            ger(M, N, (T)1, Aptr, 1, Bptr, 1, Cptr, M);
        } else if (N == 1) { // matrix * vector
            gemv('N', M, K, (T)1, Aptr, M, Bptr, 1, (T)0, Cptr, 1);
        } if (M == 1) { // rowvector * matrix
            gemv('T', K, N, (T)1, Bptr, K, Aptr, 1, (T)0, Cptr, 1);
        } else { // general case
            gemm('N', 'N', M, N, K, (T)1, Aptr, M, Bptr, K, (T)0, Cptr, M);
        }
#else /* defined(HAVE_BLAS) */
        for (int j = 1; j <= N; ++j, Cptr += M, Aptr = A.ptr()) {
            for (int k = 1; k <= K; ++k, ++Bptr, Aptr += M) {
                axpy(M, *Bptr, Aptr, 1, Cptr, 1);
            }
        }
#endif /* defined(HAVE_BLAS) */

        return res;
    }
};



} /* namespace */




// Definition
// ==================================================================
template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
fcnn::operator*(const TA &A, const TB &B)
{
    return mat_mul<TA, TB>::eval(A, B);
}


// Instantiations
// ==================================================================
INSTANTIATE_BIN(operator*)






// ==================================================================
// / (division by scalar)
// ==================================================================
namespace {

template <typename TA, typename TB>
struct div_scal {
};


template <typename T>
struct div_scal<Matrix<T>, T> {
    static Matrix<T> eval(const Matrix<T> &A, const T &B) {
        int s = A.size();
        if (!s) return A;
        Matrix<T> res(A.rows(), A.cols());
        for (int i = 1; i <= s; ++i)
            res.elem(i) = A.elem(i) / B;
        return res;
    }
};



} /* namespace */


// Definition
// ==================================================================
template <typename TA, typename TB>
typename fcnn::internal::enable_opdivpow<TA, TB>::type
fcnn::operator/(const TA &A, const TB &B)
{
    return div_scal<TA, TB>::eval(A, B);
}


// Instantiations
// ==================================================================
template
typename fcnn::internal::enable_opdivpow<Matrix<float>, float>::type
fcnn::operator/(const Matrix<float>&, const float&);
template
typename fcnn::internal::enable_opdivpow<Matrix<double>, double>::type
fcnn::operator/(const Matrix<double>&, const double&);








// ==================================================================
// pow (powers)
// ==================================================================
namespace {

template <typename TA, typename TB>
struct pow_scal {
};


template <typename T>
struct pow_scal<Matrix<T>, T> {
    static Matrix<T> eval(const Matrix<T> &A, const T &B) {
        int s = A.size();
        if (!s) return A;
        Matrix<T> res(A.rows(), A.cols());
        for (int i = 1; i <= s; ++i)
            res.elem(i) = std::pow(A.elem(i), B);
        return res;
    }
};



} /* namespace */


// Definition
// ==================================================================
template <typename TA, typename TB>
typename fcnn::internal::enable_opdivpow<TA, TB>::type
fcnn::pow(const TA &A, const TB &B)
{
    return pow_scal<TA, TB>::eval(A, B);
}


// Instantiations
// ==================================================================
template
typename fcnn::internal::enable_opdivpow<Matrix<float>, float>::type
fcnn::pow(const Matrix<float>&, const float&);
template
typename fcnn::internal::enable_opdivpow<Matrix<double>, double>::type
fcnn::pow(const Matrix<double>&, const double&);








// ==================================================================
// Matrix inverse and linear equation solving
// ==================================================================
namespace {

template <typename T>
struct inv {
};

template <typename TA, typename TB>
struct solv {
};


#if defined(HAVE_LAPACK)
extern "C" {
int
F77_FUNC(sgetrf, SGETRF)(int *M, int *N, float *A, int *lda,
                         int *ipiv, int *info);
int
F77_FUNC(dgetrf, DGETRF)(int *M, int *N, double *A, int *lda,
                         int *ipiv, int *info);
int
F77_FUNC(sgetri, SGETRI)(int *N, float *A, int *lda, int *ipiv,
                         float *work, int *lwork, int *info);
int
F77_FUNC(dgetri, DGETRI)(int *N, double *A, int *lda, int *ipiv,
                         double *work, int *lwork, int *info);
int
F77_FUNC(sgesv, SGESV)(int *N, int *Nrhs, float *A, int *lda,
                       int *ipiv, float *B, int *ldb, int *info);
int
F77_FUNC(dgesv, DGESV)(int *N, int *Nrhs, double *A, int *lda,
                       int *ipiv, double *B, int *ldb, int *info);
} /* extern "C" */



inline
void
getrf(int m, int n, float* A, int lda, int *ipiv, int *info)
{
    F77_FUNC(sgetrf, SGETRF)(&m, &n, A, &lda, ipiv, info);
}

inline
void
getrf(int m, int n, double* A, int lda, int *ipiv, int *info)
{
    F77_FUNC(dgetrf, DGETRF)(&m, &n, A, &lda, ipiv, info);
}


inline
void
getri(int N, float *A, int lda, int *ipiv, float *work, int lwork, int *info)
{
    F77_FUNC(sgetri, SGETRI)(&N, A, &lda, ipiv, work, &lwork, info);
}

inline
void
getri(int N, double *A, int lda, int *ipiv, double *work, int lwork, int *info)
{
    F77_FUNC(dgetri, DGETRI)(&N, A, &lda, ipiv, work, &lwork, info);
}


inline
void
gesv(int N, int Nrhs, float *A, int lda, int *ipiv, float *B, int ldb, int *info)
{
    F77_FUNC(sgesv, SGESV)(&N, &Nrhs, A, &lda, ipiv, B, &ldb, info);
}

inline
void
gesv(int N, int Nrhs, double *A, int lda, int *ipiv, double *B, int ldb, int *info)
{
    F77_FUNC(dgesv, DGESV)(&N, &Nrhs, A, &lda, ipiv, B, &ldb, info);
}

#endif /* defined(HAVE_LAPACK) */



template <typename T>
struct inv<Matrix<T> > {
    static Matrix<T> eval(const Matrix<T> &A) {
        int N = A.rows();
        if (N != A.cols()) {
            message mes;
            mes << "nonsquare matrix (" << N << 'x' << A.cols()
                << ") in solve";
            error(mes);
        }

        if (N == 0) return Matrix<T>(N, N);

        int info = 0;
        std::vector<int> ipiv(N);
        Matrix<T> res = A.copy();

#if defined(HAVE_LAPACK)
        getrf(N, N, res.ptr(), N, &ipiv[0], &info);
        if (info) {
            error("singular matrix in solve");
        }
        std::vector<T> work(1);
        int lwork = -1;
        getri(N, res.ptr(), N, &ipiv[0], &work[0], lwork, &info);
        lwork = (int) work[0];
        work.resize(lwork);
        getri(N, res.ptr(), N, &ipiv[0], &work[0], lwork, &info);
#else /* defined(HAVE_LAPACK) */
        error("LAPACK library not found; solve not implemented");
#endif /* defined(HAVE_LAPACK) */

        return res;
    }
};



template <typename T>
struct solv<Matrix<T>, Matrix<T> > {
    static Matrix<T> eval(const Matrix<T> &A, const Matrix<T> &B) {
        int N = A.rows(), Nrhs = B.cols();
        if (N != A.cols()) {
            message mes;
            mes << "nonsquare matrix (" << N << 'x' << A.cols()
                << ") in solve";
            error(mes);
        }
        if (N != B.rows()) {
            message mes;
            mes << "nonconformant arguments in solve; "
                << "1st operand is " << N << 'x' << N
                << ", 2nd " << B.rows() << 'x' << Nrhs;
            error(mes);
        }

        if ((N == 0) || (Nrhs == 0)) return Matrix<T>(N, Nrhs);

        int info = 0;
        std::vector<int> ipiv(N);
        Matrix<T> Acpy = A.copy();
        Matrix<T> res = B.copy();
#if defined(HAVE_LAPACK)
        gesv(N, Nrhs, Acpy.ptr(), N, &ipiv[0], res.ptr(), N, &info);
        if (info) {
            error("singular matrix in solve");
        }
#else /* defined(HAVE_LAPACK) */
        error("LAPACK library not found; solve not implemented");
#endif /* defined(HAVE_LAPACK) */

        return res;
    }
};



} /* namespace */




// Definitions
// ==================================================================
template <typename T>
typename fcnn::internal::enable_unary<T>::type
fcnn::solve(const T &A)
{
    return inv<T>::eval(A);
}

template <typename TA, typename TB>
typename fcnn::internal::enable_solve<TA, TB>::type
fcnn::solve(const TA &A, const TB &B)
{
    return solv<TA, TB>::eval(A, B);
}


// Instantiations
// ==================================================================
template
typename fcnn::internal::enable_unary<Matrix<float> >::type
fcnn::solve(const Matrix<float>&);
template
typename fcnn::internal::enable_unary<Matrix<double> >::type
fcnn::solve(const Matrix<double>&);
template
typename fcnn::internal::enable_solve<Matrix<float>, Matrix<float> >::type
fcnn::solve(const Matrix<float>&, const Matrix<float>&);
template
typename fcnn::internal::enable_solve<Matrix<double>, Matrix<double> >::type
fcnn::solve(const Matrix<double>&, const Matrix<double>&);






// ==================================================================
// stream input/output
// ==================================================================
namespace {

template <typename T>
struct out {
};

template <typename T>
struct out<Matrix<T> > {
    static std::ostream&
    put(std::ostream &os, const Matrix<T> &m)
    {
        int r = m.rows(), c = m.cols(), i, j;

        os << std::setprecision(precision<T>::val);
        for (i = 1; i <= r; i++) {
            for (j = 1; j <= c; j++) {
                os << m.elem(i, j);
                if (j < c) os << ' ';
            }
            os << '\n';
        }
        os << '\n';

        return os;
    }
};


template <typename T>
struct in {
};


template <typename T>
struct in<Matrix<T> > {
    static std::istream&
    get(std::istream &is, Matrix<T> &m)
    {
        std::vector<T> data;
        T buf;
        int ii, i, j, r = 0, c = 0, prevc = 0;

        m.reset();
        skip_all(is);

        if (is.eof()) {
            return is;
        }

        while (is) {
            if (read(is, buf)) {
                try {
                    data.push_back(buf);
                } catch (std::bad_alloc const&) {
                    error("memory allocation error");
                }
                if (c == 0) r++; // new row starts
            c++;
            }
            else if ((is.eof() && !is.fail()) && ((prevc == 0) || (c == prevc)))
                break;
            else goto err;

            if (is_eol(is)) {
                if ((prevc == 0) || (c == prevc)) {
                    prevc = c;
                    c = 0;
                }
                else goto err;

                if (is_eol(is)) {
                    c = prevc;
                    break;
                }
            }
        }

        m.reset(r, c);

        for (i = 1, ii = 0; i <= r; i++)
            for (j = 1; j <= c; j++, ii++)
                m.elem(i, j) = data[ii];

            return is;

    err:
        data.clear();
        is.clear();
        return is;
    }
};

} /* namespace */



// Definitions
// ==================================================================
template <typename T>
typename fcnn::internal::enable_io<std::ostream, T>::type&
fcnn::operator<<(std::ostream &os, const T &m)
{
    return out<T>::put(os, m);
}

template <typename T>
typename fcnn::internal::enable_io<std::istream, T>::type&
fcnn::operator>>(std::istream &is, T &m)
{
    return in<T>::get(is, m);
}



// Instantiations
// ==================================================================
template typename fcnn::internal::enable_io<std::ostream, Matrix<float> >::type&
fcnn::operator<<(std::ostream&, const Matrix<float> &m);
template typename fcnn::internal::enable_io<std::ostream, Matrix<double> >::type&
fcnn::operator<<(std::ostream&, const Matrix<double> &m);

template typename fcnn::internal::enable_io<std::istream, Matrix<float> >::type&
fcnn::operator>>(std::istream&, Matrix<float> &m);
template typename fcnn::internal::enable_io<std::istream, Matrix<double> >::type&
fcnn::operator>>(std::istream&, Matrix<double> &m);
