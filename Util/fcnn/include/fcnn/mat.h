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

/** \file mat.h
 *  \brief Floating point matrix.
 */


#ifndef FCNN_MAT_H

#define FCNN_MAT_H


#include <vector>
#include <fcnn/rcarr.h>


namespace fcnn {


/// Matrix class is used for storing input data as well as for computations
/// involving gradients and (approximate) Hessian.
template <typename T>
class Matrix
{
 public:
    /// Constructor (0x0 matrix).
    Matrix();
    /// Constructor (allocates memory).
    Matrix(int, int);
    /// Constructor (allocates memory and sets all entries to given value).
    Matrix(int, int, const T&);
    /// Constructor (allocates memory and copies values from array).
    Matrix(int, int, const T*);
    /// Copy constructor.
    Matrix(const Matrix&);

    /// Reset member function.
    Matrix& reset();
    /// Reset member function (allocates new memory).
    Matrix& reset(int, int);

    /// Destructor.
    ~Matrix() { ; };

    /// Returns pointer to data.
    inline T* ptr() { return m_data.ptr(); }
    /// Returns pointer to data (const version).
    inline T const* ptr() const { return m_data.ptr(); }

    /// Element access, no index checking.
    inline T& elem(int i, int j) { return m_data[(j - 1) * m_rows + i]; }
    /// Element access, no index checking (const version).
    inline T const& elem(int i, int j) const  {
        return m_data[(j - 1) * m_rows + i];
    }
    /// Element access with single index, no index checking.
    inline T& elem(int i)  { return m_data[i]; }
    /// Element access with single index, no index checking (const version).
    inline T const& elem(int i) const  { return m_data[i]; }

    /// Element access.
    inline T& operator()(int i) {
        if ((i < 1) || (i > m_data.size())) error_idx(i);
        return m_data[i];
    }
    /// Element access (const version).
    inline T const& operator()(int i) const {
        if ((i < 1) || (i > m_data.size())) error_idx(i);
        return m_data[i];
    }
    /// Element access.
    inline T& operator()(int i, int j) {
        if ((i < 1) || (i > m_rows) || (j < 1) || (j > m_cols))
            error_idx(i, j);
        return m_data[(j - 1) * m_rows + i];
    }
    /// Element access (const version).
    inline T const& operator()(int i, int j) const {
        if ((i < 1) || (i > m_rows) || (j < 1) || (j > m_cols))
            error_idx(i, j);
        return m_data[(j - 1) * m_rows + i];
    }

    /// Return row.
    Matrix<T> get_row(int i) const;
    /// Return submatrix created by selecting given rows.
    Matrix<T> get_rows(std::vector<int> is) const;
    /// Return column.
    Matrix<T> get_col(int j) const;
    /// Return submatrix created by selecting given columns.
    Matrix<T> get_cols(std::vector<int> js) const;
    /// Return diagonal.
    Matrix<T> get_diag() const;

    /// Returns number of rows.
    inline int rows() const { return m_rows; }
    /// Returns number of columns.
    inline int cols() const { return m_cols; }
    /// Returns size (number of elements held in memory,
    /// equal to rows * columns).
    inline int size() const { return m_data.size(); }

    /// Assignment (shallow).
    Matrix<T>& operator=(const Matrix<T>&);

    /// Return a copy of and object.
    Matrix<T> copy() const;

    /// Make unique if memory is shared with other objects.
    inline void mkunique() { m_data.mkunique(); }

 private:
    /// Data container.
    internal::rcarr<T> m_data;
    /// No. of rows.
    int m_rows;
    /// No. of columns.
    int m_cols;

    /// Report index error.
    void error_idx(int i) const;
    /// Report index error.
    void error_idx(int i, int j) const;
    /// Report row index error.
    void error_r_idx(int i) const;
    /// Report column index error.
    void error_c_idx(int j) const;

}; /* class template Matrix */


/// Create identity matrix.
template <typename T> Matrix<T> eye(int n);
/// Create matrix filled with zeros.
template <typename T>
inline
Matrix<T>
zeros(int m, int n)
{
    return Matrix<T>(m, n, T());
}
/// Create matrix filled with random numbers from uniform distribution.
template <typename T> Matrix<T> rand(int m, int n);


} /* namespace fcnn */


#endif /* FCNN_MAT_H */
