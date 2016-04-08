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

/** \file mlpnet_prune.cpp
 *  \brief Pruning algorithms for multilayer perceptron networks.
 */


#include <fcnn/mlpnet_prune.h>
#include <fcnn/mlpnet_teach.h>
#include <fcnn/matops.h>
#include <fcnn/utils.h>
#include <fcnn/level3.h>
#include <fcnn/report.h>
#include <fcnn/error.h>

#include <cmath>


#ifdef FCNN_DEBUG
#define REPORT(x) std::cerr << #x << ": " << (x) << '\n';
#endif /* FCNN_DEBUG */


using namespace fcnn;
using fcnn::internal::message;
using fcnn::internal::report;


template <typename T>
std::pair<int, int>
fcnn::mlpnet_prune_mag(MLPNet<T> &net, const Matrix<T> &in, const Matrix<T> &out,
                       T tol_level, bool report,
                       int max_reteach_iter)
{
    if (tol_level <= T()) error("tolerance level should be positive");
    T mse;
    if ((mse = net.mse(in, out)) > tol_level) {
        message mes;
        mes << "network should be trained with MSE reduced to given tolerance "
            << "level (" << tol_level << ") before pruning; MSE is " << mse;
        error(mes);
    }

    int count = 0, countn = 0;
    bool stop = false;

    while (!stop) {
        int W = net.active_w();
        Matrix<T> weights = net.get_weights();
        T minL = std::abs(weights.elem(1)), L;
        int mini = 1;
        for (int k = 2; k <= W; ++k) {
            L = std::abs(weights.elem(k));
            if (L < minL) { mini = k; minL = L; }
        }

        int wi = net.get_abs_w_idx(mini);
        net.set_active(wi, false);
        ++count;

        if (net.mse(in, out) > tol_level) {
            std::pair<T, int> retres =
                mlpnet_teach_rprop(net, in, out, tol_level, max_reteach_iter, 0);
            if (retres.first > tol_level) {
                stop = true;
                --count;
                net.set_active(wi, true);
                net.set_weights(weights);
                if (report) internal::report("pruning stopped");
            } else {
                if (report) {
                    message mes;
                    mes << "removed weight " << wi << " from " << net.total_w()
                        << " total (" << (W - 1) << " remain active); "
                        << "network has been retrained";
                    internal::report(mes);
                }
            }
        } else {
            if (report) {
                message mes;
                mes << "removed weight " << wi << " from " << net.total_w()
                    << " total (" << (W - 1) << " remain active); ";
                internal::report(mes);
            }
        }

        std::pair<int, int> rmres = net.rm_neurons(report);
        countn += rmres.first;
        count += rmres.second;
    }

    return std::pair<int, int>(count, countn);
}


template std::pair<int, int>
fcnn::mlpnet_prune_mag(MLPNet<float>&, const Matrix<float>&,
                       const Matrix<float>&,
                       float,
                       bool,
                       int);
template std::pair<int, int>
fcnn::mlpnet_prune_mag(MLPNet<double>&,
                       const Matrix<double>&, const Matrix<double>&,
                       double,
                       bool,
                       int);


template <typename T>
std::pair<int, int>
fcnn::mlpnet_prune_obs(MLPNet<T> &net, const Matrix<T> &in,
                       const Matrix<T> &out,
                       T tol_level,
                       bool report,
                       int max_reteach_iter, T alpha)
{
    if (tol_level <= T()) error("tolerance level should be positive");
    T mse;
    if ((mse = net.mse(in, out)) > tol_level) {
        message mes;
        mes << "network should be trained with MSE reduced to given tolerance "
            << "level (" << tol_level << ") before pruning; MSE is " << mse;
        error(mes);
    }

    int P = in.rows(), N = out.cols();
    T NP = (T)P * (T)N;
    int count = 0, countn = 0;
    bool stop = false;

    while (!stop) {
        int W = net.active_w();
        Matrix<T> H = ((T)1. / alpha) * eye<T>(W), grads, X, HX;
        for (int i = 1; i <= P; ++i) {
            grads = net.gradij(in, i);
#if defined(HAVE_BLAS)
            internal::ihessupdate(H.rows(), N, NP, grads.ptr(), H.ptr());
#else
            for (int j = 1; j <= N; ++j) {
                X = grads.get_col(j);
                HX = H * X;
                H = H - div(HX * t(HX), NP + t(X) * HX);
            }
#endif
        }

        Matrix<T> weights = net.get_weights();
        T L, minL = (T).5 * std::pow(weights.elem(1), (T)2.) / H.elem(1, 1);
        int mini = 1;
        for (int k = 2; k <= W; ++k) {
            L = (T).5 * std::pow(weights.elem(k), (T)2.) / H.elem(k, k);
            if (L < minL) { mini = k; minL = L; }
        }

        Matrix<T> dw = weights.elem(mini) * H.get_col(mini) / H.elem(mini, mini);
        net.set_weights(weights - dw);
        int wi = net.get_abs_w_idx(mini);
        net.set_active(wi, false);
        ++count;

        if (net.mse(in, out) > tol_level) {
            std::pair<T, int> retres =
                mlpnet_teach_rprop(net, in, out, tol_level, max_reteach_iter, 0);
            if (retres.first > tol_level) {
                stop = true;
                --count;
                net.set_active(wi, true);
                net.set_weights(weights);
                if (report) internal::report("pruning stopped");
            } else {
                if (report) {
                    message mes;
                    mes << "removed weight " << wi << " from " << net.total_w()
                        << " total (" << (W - 1) << " remain active); "
                        << "network has been retrained";
                    internal::report(mes);
                }
            }
        } else {
            if (report) {
                message mes;
                mes << "removed weight " << wi << " from " << net.total_w()
                    << " total (" << (W - 1) << " remain active); ";
                internal::report(mes);
            }
        }

        std::pair<int, int> rmres = net.rm_neurons(report);
        countn += rmres.first;
        count += rmres.second;
    }

    return std::pair<int, int>(count, countn);
}





template std::pair<int, int>
fcnn::mlpnet_prune_obs(MLPNet<float>&, const Matrix<float>&,
                       const Matrix<float>&,
                       float,
                       bool,
                       int,
                       float);
template std::pair<int, int>
fcnn::mlpnet_prune_obs(MLPNet<double>&,
                       const Matrix<double>&, const Matrix<double>&,
                       double,
                       bool,
                       int,
                       double);



