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

/** \file matops.h
 *  \brief Matrix operations.
 */


#ifndef FCNN_MATOPS_H

#define FCNN_MATOPS_H


#include <iostream>
#include <fcnn/mat.h>
#include <fcnn/matops_enable.h>


namespace fcnn {


/// Unary minus.
template <typename T>
typename fcnn::internal::enable_unary<T>::type
operator-(const T&);

/// Transposition.
template <typename T>
typename fcnn::internal::enable_unary<T>::type
t(const T&);

/// Reshape.
template <typename T>
typename fcnn::internal::enable_unary<T>::type
reshape(const T&, int m, int n);

/// Addition.
template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
operator+(const TA&, const TB&);

/// Subtraction.
template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
operator-(const TA&, const TB&);

/// Multiplication.
template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
operator*(const TA&, const TB&);

/// Division (by scalar).
template <typename TA, typename TB>
typename fcnn::internal::enable_opdivpow<TA, TB>::type
operator/(const TA&, const TB&);

/// Element by element power.
template <typename TA, typename TB>
typename fcnn::internal::enable_opdivpow<TA, TB>::type
pow(const TA&, const TB&);

/// Element by element multiplication.
template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
mul(const TA&, const TB&);

/// Element by element division.
template <typename TA, typename TB>
typename fcnn::internal::enable_binary<TA, TB>::type
div(const TA&, const TB&);

/// Matrix inverse.
template <typename T>
typename fcnn::internal::enable_unary<T>::type
solve(const T&);

/// Solve linear equation.
template <typename TA, typename TB>
typename fcnn::internal::enable_solve<TA, TB>::type
solve(const TA&, const TB&);

/// Stream output.
template <typename T>
typename fcnn::internal::enable_io<std::ostream, T>::type&
operator<<(std::ostream&, const T&);

/// Stream input.
template <typename T>
typename fcnn::internal::enable_io<std::istream, T>::type&
operator>>(std::istream&, T&);


} /* namespace fcnn */


#endif /* FCNN_MATOPS_H */
