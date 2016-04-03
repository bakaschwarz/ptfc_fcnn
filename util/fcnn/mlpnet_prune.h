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

/** \file mlpnet_prune.h
 *  \brief Pruning algorithms for multilayer perceptron networks.
 */


#ifndef FCNN_MPLNET_PRUNE_H

#define FCNN_MPLNET_PRUNE_H


#include <mlpnet.h>


namespace fcnn {


/// Minimum magnitude pruning. Returns no. of deleted weights and neurons.
/// Parameter max_reteach_iter determines maximum no. of iterations while
/// reteaching network. When this number is reached and tol_level
/// is not achieved, pruning stops and last turned off weight is turned back on.
template <typename T>
std::pair<int, int>
mlpnet_prune_mag(MLPNet<T> &net, const Matrix<T> &in, const Matrix<T> &out,
                 T tol_level, bool report = false,
                 int max_reteach_iter = 50);


/// Minimum magnitude pruning. Returns no. of deleted weights and neurons.
/// Parameter max_reteach_iter determines maximum no. of iterations while
/// reteaching network. When this number is reached and tol_level
/// is not achieved, pruning stops and last turned off weight is turned back on.
template <typename T>
inline
std::pair<int, int>
mlpnet_prune_mag(MLPNet<T> &net, const Dataset<T> &dat,
                 T tol_level, bool report = false,
                 int max_reteach_iter = 50)
{
    return mlpnet_prune_mag(net, dat.get_input(), dat.get_output(),
                            tol_level, report,
                            max_reteach_iter);
}


/// Optimal Brain Surgeon. Returns no. of deleted weights and neurons.
/// Parameter alpha is used in Hessian approximation. According to paper,
/// this should be between 1e-8 and 1e-4. Parameter max_reteach_iter
/// determines maximum no. of iterations while reteaching network. When
/// this number is reached and tol_level is not achieved, pruning stops
/// and last turned off weight is turned back on.
template <typename T>
std::pair<int, int>
mlpnet_prune_obs(MLPNet<T> &net, const Matrix<T> &in, const Matrix<T> &out,
                 T tol_level, bool report = false,
                 int max_reteach_iter = 10, T alpha = (T)1e-5);

/// Optimal Brain Surgeon. Returns no. of deleted weights and neurons.
/// Parameter alpha is used in Hessian approximation. According to paper,
/// this should be between 1e-8 and 1e-4. Parameter max_reteach_iter
/// determines maximum no. of iterations while reteaching network. When
/// this number is reached and tol_level is not achieved, pruning stops
/// and last turned off weight is turned back on.
template <typename T>
inline
std::pair<int, int>
mlpnet_prune_obs(MLPNet<T> &net, const Dataset<T> &dat,
                 T tol_level, bool report = false,
                 int max_reteach_iter = 10, T alpha = (T)1e-5)
{
    return mlpnet_prune_obs(net, dat.get_input(), dat.get_output(),
                            tol_level, report,
                            max_reteach_iter, alpha);
}



} /* namespace fcnn */


#endif /* FCNN_MPLNET_PRUNE_H */
