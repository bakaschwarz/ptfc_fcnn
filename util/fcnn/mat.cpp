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

/** \file mat.cpp
 *  \brief Floating point matrix.
 */


#include <mat.h>
#include <error.h>
#include <utils.h>
#include <level1.h>
#include <cstdlib>
#include <algorithm>



using namespace fcnn;
using namespace fcnn::internal;



// Constructors
template <typename T>
Matrix<T>::Matrix()
{
    m_rows = m_cols = 0;
}


template <typename T>
Matrix<T>::Matrix(int r, int c)
{
    if ((r < 0) || (c < 0)) error("negative size");
    m_rows = r;
    m_cols = c;
    m_data.reset(m_rows * m_cols);
}


template <typename T>
Matrix<T>::Matrix(int r, int c, const T& n)
{
    if ((r < 0) || (c < 0)) error("negative size");
    m_rows = r;
    m_cols = c;
    m_data.reset(m_rows * m_cols);
    m_data.set_all_to(n);
}


template <typename T>
Matrix<T>::Matrix(int r, int c, const T *arr)
{
    if ((r < 0) || (c < 0)) error("negative size");
    m_rows = r;
    m_cols = c;
    m_data.reset(m_rows * m_cols);
    m_data.read_from(arr);
}


template <typename T>
Matrix<T>::Matrix(const Matrix<T> &mat) : m_data(mat.m_data)
{
    m_rows = mat.m_rows;
    m_cols = mat.m_cols;
}


// Reset function
template <typename T>
Matrix<T>&
Matrix<T>::reset()
{
    m_rows = m_cols = 0;
    m_data.reset();

    return *this;
}


template <typename T>
Matrix<T>&
Matrix<T>::reset(int r, int c)
{
    if ((r < 0) || (c < 0)) error("negative size");
    m_rows = r;
    m_cols = c;
    m_data.reset(m_rows * m_cols);

    return *this;
}


// Assignments
template <typename T>
Matrix<T>&
Matrix<T>::operator=(const Matrix<T> &mat)
{
    m_rows = mat.m_rows;
    m_cols = mat.m_cols;
    m_data = mat.m_data;

    return *this;
}


// Copy
template <typename T>
Matrix<T>
Matrix<T>::copy() const
{
    Matrix<T> res = *this;
    res.mkunique();
    return res;
}


// Rows and columns
template <typename T>
Matrix<T>
Matrix<T>::get_row(int i) const
{
    if ((i < 1) || (i > m_rows)) error_r_idx(i);
    Matrix<T> res(1, m_cols);
    internal::copy(m_cols, ptr() + i - 1, m_rows, res.ptr(), 1);
    return res;
}


template <typename T>
Matrix<T>
Matrix<T>::get_rows(std::vector<int> is) const
{
    int nr = is.size();
    Matrix<T> res(nr, m_cols);
    for (int i = 0; i < nr; ++i) {
        int ii = is[i];
        if ((ii < 1) || (ii > m_rows)) error_r_idx(ii);
        internal::copy(m_cols, ptr() + ii - 1, m_rows, res.ptr() + i, nr);
    }
    return res;
}


template <typename T>
Matrix<T>
Matrix<T>::get_col(int j) const
{
    if ((j < 1) || (j > m_cols)) error_c_idx(j);
    Matrix<T> res(m_rows, 1);
    internal::copy(m_rows, ptr() + (j - 1) * m_rows, 1, res.ptr(), 1);
    return res;
}


template <typename T>
Matrix<T>
Matrix<T>::get_cols(std::vector<int> js) const
{
    int nc = js.size();
    Matrix<T> res(m_rows, nc);
    for (int j = 0; j < nc; ++j) {
        int jj = js[j];
        if ((jj < 1) || (jj > m_cols)) error_c_idx(jj);
        internal::copy(m_rows, ptr() + (jj - 1) * m_rows, 1, res.ptr() + j * m_rows, 1);
    }
    return res;
}


template <typename T>
Matrix<T>
Matrix<T>::get_diag() const
{
    int n = std::min(m_rows, m_cols);
    Matrix<T> res(n, 1, T());
    for (int i = 1; i <= n; ++i) {
        res.elem(i) = elem(i, i);
    }
    return res;
}


// Indexing erros
template <typename T>
void
Matrix<T>::error_idx(int i) const
{
    message mes;
    mes << "invalid single index " << i << "; matrix size: " << m_data.size();
    error(mes);
}


template <typename T>
void
Matrix<T>::error_idx(int i, int j) const
{
    message mes;
    mes << "invalid double index (" << i << ", " << j << "); matrix size: "
        << m_rows << 'x' << m_cols;
    error(mes);
}


template <typename T>
void
Matrix<T>::error_r_idx(int i) const
{
    message mes;
    mes << "invalid row index (" << i << "); matrix size: "
        << m_rows << 'x' << m_cols;
    error(mes);
}

template <typename T>
void
Matrix<T>::error_c_idx(int j) const
{
    message mes;
    mes << "invalid column index (" << j << "); matrix size: "
        << m_rows << 'x' << m_cols;
    error(mes);
}



// Instantiation
template class Matrix<double>;
template class Matrix<float>;




template <typename T>
Matrix<T>
fcnn::eye(int n)
{
    Matrix<T> res(n, n, T());
    int i, np1 = n + 1, ii;
    T *p = res.ptr();
    for (i = 1, ii = 0; i <= n; ++i, ii += np1) p[ii] = (T)1.;
    return res;
}


template Matrix<float> fcnn::eye(int);
template Matrix<double> fcnn::eye(int);


template <typename T>
Matrix<T>
fcnn::rand(int m, int n)
{
    Matrix<T> res(m, n);
    int i, mn = m * n;
    T *p = res.ptr();
    for (i = 0; i < mn; ++i) p[i] = (T) ::rand() / (T) RAND_MAX;
    return res;
}


template Matrix<float> fcnn::rand(int, int);
template Matrix<double> fcnn::rand(int, int);






















