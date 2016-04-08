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

/** \file mlpnet_teach.h
 *  \brief Teaching algorithms for multilayer perceptron networks.
 */


#ifndef FCNN_MPLNET_TEACH_H

#define FCNN_MPLNET_TEACH_H


#include <fcnn/mlpnet.h>


namespace fcnn {


/// Standard batch backpropagation algorithm. Returns the final MSE and the number
/// of iterations. Safe choice of learning rate is 0.7.
template <typename T>
std::pair<T, int>
mlpnet_teach_bp(MLPNet<T> &net, const Matrix<T> &in, const Matrix<T> &out,
                T tol_level, int max_epochs, T learn_rate, int report_freq = 0,
                T l2reg = T());
/// Standard batch backpropagation algorithm. Returns the final MSE and the number
/// of iterations. Safe choice of learning rate is 0.7.
template <typename T>
inline
std::pair<T, int>
mlpnet_teach_bp(MLPNet<T> &net, const Dataset<T> &dat,
                T tol_level, int max_epochs, T learn_rate, int report_freq = 0,
                T l2reg = T())
{
    return mlpnet_teach_bp(net, dat.get_input(), dat.get_output(),
                           tol_level, max_epochs, learn_rate, report_freq, l2reg);
}



/// Rprop algorithm (batch). Returns the final MSE and the number
/// of iterations. Safe choices of parameters are: u = 1.2, d = 0.5,
/// gmax = 50. and gmin = 1e-6.
template <typename T>
std::pair<T, int>
mlpnet_teach_rprop(MLPNet<T> &net, const Matrix<T> &in, const Matrix<T> &out,
                   T tol_level, int max_epochs, int report_freq = 0, T l2reg = T(),
                   T u = (T)1.2, T d = (T)0.5, T gmax = (T)50., T gmin = 1e-6);

/// Rprop algorithm (batch). Returns the final MSE and the number
/// of iterations. Safe choices of parameters are: u = 1.2, d = 0.5,
/// gmax = 50. and gmin = 1e-6.
template <typename T>
inline
std::pair<T, int>
mlpnet_teach_rprop(MLPNet<T> &net, const Dataset<T> &dat,
                   T tol_level, int max_epochs, int report_freq = 0, T l2reg = T(),
                   T u = (T)1.2, T d = (T)0.5, T gmax = (T)50., T gmin = 1e-6)
{
    return mlpnet_teach_rprop(net, dat.get_input(), dat.get_output(),
                              tol_level, max_epochs, report_freq, l2reg,
                              u, d, gmax, gmin);
}



/// Stochastic gradient descent with (optional) RMS weights scaling, weight
/// decay, and momentum. Safe choices of parameters are: minibatch size = 100,
/// lambda = 0.1 (rmsprop parameter controlling the update of mean squared gradient),
/// gamma (weight decay parameter) = 0, momentum (momentum parameter) = 0.5.
template <typename T>
std::pair<T, int>
mlpnet_teach_sgd(MLPNet<T> &net, const Matrix<T> &in, const Matrix<T> &out,
                 T tol_level, int max_epochs, T learn_rate, int report_freq = 0, T l2reg = T(),
                 int minibatchsz = 100, T lambda = 0.1, T gamma = 0, T momentum = 0.5);


/// Stochastic gradient descent with (optional) RMS weights scaling, weight
/// decay, and momentum. Safe choices of parameters are: minibatch size = 100,
/// lambda = 0.1 (rmsprop parameter controlling the update of mean squared gradient),
/// gamma (weight decay parameter) = 0, momentum (momentum parameter) = 0.5.
template <typename T>
inline
std::pair<T, int>
mlpnet_teach_sgd(MLPNet<T> &net, const Dataset<T> &dat,
                 T tol_level, int max_epochs, T learn_rate, int report_freq = 0, T l2reg = T(),
                 int minibatchsz = 100, T lambda = 0.1, T gamma = 0, T momentum = 0.5)
{
    return mlpnet_teach_sgd(net, dat.get_input(), dat.get_output(),
                            tol_level, max_epochs, learn_rate, report_freq, l2reg,
                            minibatchsz, lambda, gamma, momentum);
}



} /* namespace fcnn */


#endif /* FCNN_MPLNET_TEACH_H */
