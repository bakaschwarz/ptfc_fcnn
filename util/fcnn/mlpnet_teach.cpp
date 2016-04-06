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

/** \file mlpnet_teach.cpp
 *  \brief Teaching algorithms for multilayer perceptron networks.
 */


#include <mlpnet_teach.h>
#include <matops.h>
#include <utils.h>
#include <report.h>
#include <error.h>
#include <algorithm>


using namespace fcnn;
using fcnn::internal::message;
using fcnn::internal::report;
using fcnn::internal::sample_int;


template <typename T>
std::pair<T, int>
fcnn::mlpnet_teach_bp(MLPNet<T> &net,
                      const Matrix<T> &in, const Matrix<T> &out,
                      T tol_level, int max_epochs, T learn_rate, int report_freq,
                      T l2reg)
{
    if (tol_level <= T()) error("tolerance level should be positive");
    if (learn_rate <= T()) error("learning rate should be positive");
    if (l2reg < T()) error("L2 regularization parameter should be nonnegative");
    int i = 0;
    T mse;
    Matrix<T> w0, w1, g;
    std::pair<Matrix<T>, T> gm = net.grad(in, out);
    g = gm.first;
    mse = gm.second;
    if (mse < tol_level) return std::pair<T, int>(mse, i);
    w0 = net.get_weights();

    for (++i; i <= max_epochs; ++i) {
        // update
        if (l2reg != T()) g = g + l2reg * w0;
        w1 = w0 - learn_rate * g;
        net.set_weights(w1);
        // gradient, mse
        gm = net.grad(in, out);
        g = gm.first;
        mse = gm.second;
        if (report_freq) {
            if (i && !(i % report_freq)) {
                message mes;
                mes << "backpropagation; epoch " << i << ", mse: " << mse
                    << " (desired: " << tol_level << ")";
                report(mes);
            }
        }
        if (mse < tol_level) break;
        w0 = w1;
    }
    if (i > max_epochs) --i;
    return std::pair<T, int>(mse, i);
}



template std::pair<float, int>
fcnn::mlpnet_teach_bp(MLPNet<float>&,
                      const Matrix<float>&, const Matrix<float>&,
                      float, int, float, int,
                      float);
template std::pair<double, int>
fcnn::mlpnet_teach_bp(MLPNet<double>&,
                      const Matrix<double>&, const Matrix<double>&,
                      double, int, double, int,
                      double);






template <typename T>
std::pair<T, int>
fcnn::mlpnet_teach_rprop(MLPNet<T> &net,
                         const Matrix<T> &in, const Matrix<T> &out,
                         T tol_level, int max_epochs, int report_freq, T l2reg,
                         T u, T d, T gmax, T gmin)
{
    if (tol_level <= T()) error("tolerance level should be positive");
    if (l2reg < T()) error("L2 regularization parameter should be nonnegative");
    int i = 0;
    T mse;
    Matrix<T> w0, w1, g0, g1, gamma, dw;

    // init
    std::pair<Matrix<T>, T> gm = net.grad(in, out);
    g0 = gm.first;
    mse = gm.second;
    if (mse < tol_level) return std::pair<T, int>(mse, i);
    w0 = net.get_weights();
    if (l2reg != T()) g0 = g0 + l2reg * w0;
    w1 = w0 - (T)0.7 * g0;
    net.set_weights(w1);
    w0 = w1;

    // init (2nd gradient)
    ++i;
    gm = net.grad(in, out);
    g1 = gm.first;
    mse = gm.second;
    if (report_freq) {
        if (!(i % report_freq)) {
            message mes;
            mes << "Rprop; epoch " << i << ", mse: " << mse << " (desired: "
                << tol_level << ")";
            report(mes);
        }
    }
    if (mse < tol_level) return std::pair<T, int>(mse, i);

    // init gamma and dw (step) vectors
    int N = w0.rows();
    gamma = Matrix<T>(N, 1, (T)(gmin > 1e-1) ?
                            gmin : ((gmax > 1e-1) ? 1e-1 : gmax));
    dw = Matrix<T>(N, 1, (T)0);

    for (++i; i <= max_epochs; ++i) {
        // determine step and update gamma
        for (int n = 1; n <= N; ++n) {
            if (g0.elem(n) * g1.elem(n) > 0) {
                if (g1.elem(n) > 0) dw.elem(n) = -gamma.elem(n);
                else dw.elem(n) = gamma.elem(n);
                gamma.elem(n) = std::min(u * gamma.elem(n), gmax);
            } else if (g0.elem(n) * g1.elem(n) < 0) {
                dw.elem(n) = 0;
                gamma.elem(n) = std::max(d * gamma.elem(n), gmin);
            } else {
                if (g1.elem(n) > 0) dw.elem(n) = -gamma.elem(n);
                else if (g1.elem(n) < 0) dw.elem(n) = gamma.elem(n);
                else dw.elem(n) = 0;
            }
        }
        // update
        w1 = w0 + dw;
        net.set_weights(w1);
        // next gradients
        g0 = g1;
        gm = net.grad(in, out);
        g1 = gm.first;
        if (l2reg != T()) g1 = g1 + l2reg * w1;
        mse = gm.second;
        if (report_freq) {
            if (i && !(i % report_freq)) {
                message mes;
                mes << "Rprop; epoch " << i
                    << ", mse: " << mse << " (desired: "
                    << tol_level << ")";
                report(mes);
            }
        }
        if (mse < tol_level) break;
        w0 = w1;
    }
    if (i > max_epochs) --i;
    return std::pair<T, int>(mse, i);
}




template std::pair<float, int>
fcnn::mlpnet_teach_rprop(MLPNet<float>&,
                         const Matrix<float>&, const Matrix<float>&,
                         float, int, int,
                         float, float, float, float, float);
template std::pair<double, int>
fcnn::mlpnet_teach_rprop(MLPNet<double>&,
                         const Matrix<double>&, const Matrix<double>&,
                         double, int, int,
                         double, double, double, double, double);






template <typename T>
std::pair<T, int>
fcnn::mlpnet_teach_sgd(MLPNet<T> &net, const Matrix<T> &in, const Matrix<T> &out,
                       T tol_level, int max_epochs, T learn_rate, int report_freq, T l2reg,
                       int minibatchsz, T lambda, T gamma, T momentum)
{
    if (tol_level <= T()) error("tolerance level should be positive");
    if (learn_rate <= T()) error("learning rate should be positive");
    if (l2reg < T()) error("L2 regularization parameter should be nonnegative");
    int i = 0, N = in.rows(), M = minibatchsz, W = net.active_w();
    T mse;
    Matrix<T> w0, w1, dw, ms, mm, g;
    std::pair<Matrix<T>, T> gm;
    std::vector<int> idx;

    if ((M < 1) || (M >= N)) {
        error("minibatch size should be at least 1 and less than the number of records");
    }
    if (lambda != T()) {
        ms = Matrix<T>(W, 1, (T)1);
    }
    if (momentum != T()) {
        mm = Matrix<T>(W, 1, T());
    }
    idx = sample_int(N, M);
    gm = net.grad(in.get_rows(idx), out.get_rows(idx));
    g = gm.first;
    w0 = net.get_weights();
    if (l2reg != T()) g = g + l2reg * w0;
    mse = gm.second;
    if (mse < tol_level) {
        mse = net.mse(in, out);
        if (mse < tol_level) {
            return std::pair<T, int>(mse, i);
        }
    }

    bool mseall = false;
    for (++i; i <= max_epochs; ++i) {
        dw = -learn_rate * g;
        if (lambda != T()) {
            dw = div(dw, pow(ms, (T).5));
            ms = (1 - lambda) * ms + lambda * pow(g, (T)2);
        }
        if (gamma != T()) dw = dw / (1 + gamma * (i - 1));
        if (momentum != T()) {
            dw = momentum * mm + dw;
            mm = dw;
        }
        w1 = w0 + dw;
        net.set_weights(w1);
        idx = sample_int(N, M);
        gm = net.grad(in.get_rows(idx), out.get_rows(idx));
        g = gm.first;
        if (l2reg != T()) g = g + l2reg * w1;
        mse = gm.second;
        mseall = false;
        if (report_freq) {
            if (i && !(i % report_freq)) {
                message mes;
                mes << "stochastic gradient descent; epoch " << i << ", mse: "
                    << mse << " (desired: " << tol_level << ")";
                report(mes);
            }
        }
        if (mse < tol_level) {
            mse = net.mse(in, out);
            mseall = true;
            if (mse < tol_level) break;
        }
        w0 = w1;
    }
    if (!mseall) mse = net.mse(in, out);

    if (i > max_epochs) --i;
    return std::pair<T, int>(mse, i);
}



template std::pair<float, int>
fcnn::mlpnet_teach_sgd(MLPNet<float>&, const Matrix<float>&, const Matrix<float>&,
                       float, int, float, int, float,
                       int, float, float, float);
template std::pair<double, int>
fcnn::mlpnet_teach_sgd(MLPNet<double>&, const Matrix<double>&, const Matrix<double>&,
                       double, int, double, int, double,
                       int, double, double, double);



