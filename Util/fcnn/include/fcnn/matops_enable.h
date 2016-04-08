/*
 *  This file is a part of Fast Compressed Neural Networks.
 *
 *  Copyright (c) Grzegorz Klima 2012-2016
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

/** \file matops_enable.h
 *  \brief Matrix operations 'enablers'.
 */


#ifndef FCNN_MATOPS_ENABLE_H

#define FCNN_MATOPS_ENABLE_H


#include <fcnn/mat.h>
#include <iostream>


namespace fcnn {
namespace internal {


template <typename T>
struct allow_scalar {
    static const bool val = false;
};
template <>
struct allow_scalar<float> {
    static const bool val = true;
};
template <>
struct allow_scalar<double> {
    static const bool val = true;
};


template <typename TA, typename TB>
struct is_equal {
    static const bool val = false;
};
template <typename T>
struct is_equal<T, T> {
    static const bool val = true;
};


template <typename T>
struct is_matrix {
    static const bool val = false;
};
template <typename T>
struct is_matrix<Matrix<T> > {
    static const bool val = true;
};


template <typename T>
struct get_scalar {
    typedef T type;
};
template <typename T>
struct get_scalar<Matrix<T> > {
    typedef T type;
};


template <typename T, bool> struct enable_unary_ { };
template <typename T>
struct enable_unary_<Matrix<T>, true> {
    typedef Matrix<T> type;
};

template <typename T>
struct enable_unary : enable_unary_<T,
        allow_scalar<typename get_scalar<T>::type>::val> {
};


template <typename TA, typename TB, bool> struct enable_binary_ { };
template <typename TA, typename TB>
struct enable_binary_<TA, TB, true> {
    typedef Matrix<typename get_scalar<TA>::type> type;
};

template <typename TA, typename TB>
struct enable_binary : enable_binary_<TA, TB,
        ((allow_scalar<typename get_scalar<TA>::type>::val)
        && (is_equal<typename get_scalar<TA>::type,
                    typename get_scalar<TB>::type>::val)
        && (is_matrix<TA>::val || is_matrix<TB>::val))> {
};


template <typename TA, typename TB, bool> struct enable_solve_ { };
template <typename TA, typename TB>
struct enable_solve_<TA, TB, true> {
    typedef Matrix<typename get_scalar<TA>::type> type;
};

template <typename TA, typename TB>
struct enable_solve : enable_solve_<TA, TB,
        ((allow_scalar<typename get_scalar<TA>::type>::val)
        && (is_equal<typename get_scalar<TA>::type,
                    typename get_scalar<TB>::type>::val)
        && (is_matrix<TA>::val && is_matrix<TB>::val))> {
};


template <typename TA, typename TB, bool> struct enable_opdivpow_ { };
template <typename TA, typename TB>
struct enable_opdivpow_<TA, TB, true> {
    typedef TA type;
};

template <typename TA, typename TB>
struct enable_opdivpow : enable_opdivpow_<TA, TB,
        ((is_matrix<TA>::val)
        && (!is_matrix<TB>::val)
        && (allow_scalar<typename get_scalar<TA>::type>::val)
        && (is_equal<typename get_scalar<TA>::type,
                    typename get_scalar<TB>::type >::val))> {
};


template <typename TA, typename TB, bool ok>
struct enable_io_ { };
template <typename TA, typename TB>
struct enable_io_<TA, TB, true> {
    typedef TA type;
};

template <typename TA, typename TB>
struct enable_io : enable_io_<TA, TB,
        ((is_matrix<TB>::val)
        && (allow_scalar<typename get_scalar<TB>::type>::val))> {
};


} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_MATOPS_ENABLE_H */
