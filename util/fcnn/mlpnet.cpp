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

/** \file mlpnet.cpp
 *  \brief Multilayer perceptron network.
 */

#include <error.h>
#include <utils.h>
#include <mlpnet.h>
#include <struct.h>
#include <export.h>
#include <level3.h>
#include <activation.h>
#include <matops.h>
#include <iostream>
#include <cstdlib>
#include <cstdarg>


using namespace fcnn;
using namespace fcnn::internal;


// ==================================================================
// Dump on std::cerr
// ==================================================================
#ifdef FCNN_DEBUG
namespace {

template <typename T>
void
dump(const std::string &s, const std::vector<T> &v)
{
    int n = v.size();
    std::cerr << s << ":\n";
    for (int i = 0; i < n; ++i) std::cerr << ' ' << v[i];
    std::cerr << '\n';
}


} /* namespace */



template <typename T>
void
MLPNet<T>::dump() const
{
    std::cerr << "name:\n \"" << m_name << "\"\n";
    if (m_nol) {
        ::dump("layers", m_l);
        ::dump("neuron pointers", m_n_p);
        ::dump("connections to prev layer", m_n_prev);
        ::dump("connections to next layer", m_n_next);
        ::dump("weight pointers", m_w_p);
        ::dump("weight values", m_w_val);
        ::dump("weight flags", m_w_fl);
        std::cerr << "total weights:\n " << total_w() << "\nactive weights:\n "
                << active_w() << "\n\n";
        std::cerr << "activation functions:\n";
        for (int i = 1; i < m_af.size(); ++i)
            std::cerr << " l" << (i + 1) << ": " << m_af[i] << " (" << m_af_p[i] << ')';
        std::cerr << '\n';
    }
}


#define REPORT(x) std::cerr << #x << ": " << (x) << '\n';

#endif /* FCNN_DEBUG */




// ==================================================================
// Construction, destruction, reconstruction
// ==================================================================
template <typename T>
MLPNet<T>::MLPNet(const std::vector<int> &layers)
{
    construct(layers);
}


template <typename T>
MLPNet<T>::MLPNet(int no_of_layers, ...)
{
    va_list arg;
    va_start(arg, no_of_layers);
    std::vector<int> layers(no_of_layers);
    for(int i = 0; i < no_of_layers; i++) {
        layers[i] = va_arg(arg, int);
    }
    va_end(arg);
    construct(layers);
}


template <typename T>
void
MLPNet<T>::construct(const std::vector<int> &layers)
{
    m_l = layers;
    m_nol = m_l.size();
    try {
        mlp_construct(m_l, m_n_p, m_n_prev, m_n_next,
                      m_w_p, m_w_val, m_w_fl, m_w_on);
    } catch (exception &e) {
        error(e.what());
    }
    m_af.assign(m_nol, sym_sigmoid); m_af[0] = 0;
    m_af_p.assign(m_nol, mlp_act_f_pdefault<T>(sym_sigmoid)); m_af_p[0] = (T)0;
}



template <typename T>
void
MLPNet<T>::construct(const std::vector<int> &layers,
                     const std::vector<int> &w_active,
                     const std::vector<T> &w_vals)
{
    m_l = layers;
    m_nol = m_l.size();
    m_w_fl = w_active;
    try {
        mlp_construct(m_l, m_n_p, m_n_prev, m_n_next,
                      m_w_p, w_vals, m_w_val, m_w_fl, m_w_on);
    } catch (exception &e) {
        error(e.what());
    }
    m_af.assign(m_nol, sym_sigmoid); m_af[0] = 0;
    m_af_p.assign(m_nol, mlp_act_f_pdefault<T>(sym_sigmoid)); m_af_p[0] = (T)0;
}


template <typename T>
void
MLPNet<T>::expand_reorder_inputs(int newnoinp, const std::map<int, int> &m)
{
    try {
        mlp_expand_reorder_inputs(m_l, m_n_p, m_n_prev, m_n_next,
                                  m_w_p, m_w_val, m_w_fl,
                                  newnoinp, m);
    } catch (exception &e) {
        error(e.what());
    }
}



template <typename T>
std::pair<int, int>
MLPNet<T>::rm_neurons(bool report)
{
    int w = m_w_on, n;
    n = mlp_rm_neurons(m_l, m_n_p, m_n_prev, m_n_next,
                       m_w_p, m_w_val, m_w_fl, m_w_on,
                       m_af, m_af_p,
                       report);
    w -= m_w_on;
    return std::pair<int, int>(n, w);
}



template <typename T>
std::vector<int>
MLPNet<T>::rm_input_neurons(bool report)
{
    std::vector<int> ind;
    ind.reserve(m_l[0]);
    for (int i = 0; i < m_l[0]; ++i)
        if (m_n_next[i]) ind.push_back(i + 1);
    mlp_rm_input_neurons(m_l, m_n_p, m_n_prev, m_n_next,
                         m_w_p, m_w_val, m_w_fl,
                         report);
    return ind;
}




template <typename T>
void
MLPNet<T>::clear()
{
    *this = MLPNet();
}



template <typename T>
MLPNet<T>
fcnn::merge(const MLPNet<T> &A, const MLPNet<T> &B, bool same_inputs)
{
    MLPNet<T> res;

    try {
        mlp_merge(A.m_l, A.m_w_p, A.m_w_val, A.m_w_fl,
                B.m_l, B.m_w_p, B.m_w_val, B.m_w_fl,
                same_inputs,
                res.m_l, res.m_n_p,
                res.m_n_prev, res.m_n_next,
                res.m_w_p, res.m_w_val,
                res.m_w_fl, res.m_w_on);
    } catch (exception &e) {
        error(e.what());
    }
    res.m_nol = res.m_l.size();
    for (int l = 1; l < res.m_nol; ++l) {
        if ((A.m_af[l] != B.m_af[l]) || (A.m_af_p[l] != B.m_af_p[l]))
            error("activation functions in networks disagree");
    }
    res.m_af = A.m_af;
    res.m_af_p = A.m_af_p;
    return res;
}


template MLPNet<float> fcnn::merge(const MLPNet<float>&, const MLPNet<float>&, bool);
template MLPNet<double> fcnn::merge(const MLPNet<double>&, const MLPNet<double>&, bool);



template <typename T>
MLPNet<T>
fcnn::stack(const MLPNet<T> &A, const MLPNet<T> &B)
{
    MLPNet<T> res;

    try {
        mlp_stack(A.m_l, A.m_w_p, A.m_w_val, A.m_w_fl,
                B.m_l, B.m_w_p, B.m_w_val, B.m_w_fl,
                res.m_l, res.m_n_p,
                res.m_n_prev, res.m_n_next,
                res.m_w_p, res.m_w_val,
                res.m_w_fl, res.m_w_on);
    } catch (exception &e) {
        error(e.what());
    }
    res.m_nol = res.m_l.size();
    res.m_af = A.m_af;
    res.m_af.insert(res.m_af.end(), B.m_af.begin() + 1, B.m_af.end());
    res.m_af_p = A.m_af_p;
    res.m_af_p.insert(res.m_af_p.end(), B.m_af_p.begin() + 1, B.m_af_p.end());
    return res;
}


template MLPNet<float> fcnn::stack(const MLPNet<float>&, const MLPNet<float>&);
template MLPNet<double> fcnn::stack(const MLPNet<double>&, const MLPNet<double>&);


// ==================================================================
// Activation functions
// ==================================================================
template <typename T>
void
MLPNet<T>::set_act_f(int l, int af, T param)
{
    if ((l < 2) || (l > m_nol)) {
        error("invalid layer index");
    }
    if (!mlp_act_f_valid(af))
        error("invalid activation function");
    if (param < T()) error("activation function parameter must be positive");
    if (param == T()) param = mlp_act_f_pdefault<T>(af);
    m_af[l - 1] = af;
    m_af_p[l - 1] = param;
}




// ==================================================================
// Indexing layers, neurons and weights
// ==================================================================
template <typename T>
void
MLPNet<T>::check_l(int l) const
{
    if ((l < 1) || (l > m_nol)) {
        message mes;
        mes << "invalid layer index: " << l << " (number of layers is"
            << m_nol << ")";
        error(mes);
    }
}


template <typename T>
int
MLPNet<T>::no_neurons(int l) const
{
    check_l(l);
    return m_l[l - 1];
}



template <typename T>
void
MLPNet<T>::check_n(int l, int n) const
{
    if ((l < 1) || (l > m_nol)) {
        message mes;
        mes << "invalid layer index: " << l << " (number of layers is"
            << m_nol << ")";
        error(mes);
    }
    if ((n < 1) || (n > m_l[l - 1])) {
        message mes;
        mes << "invalid neuron index: " << n << " in layer " << l
            << " (there are " << m_l[l - 1] << " neurons in this layer)";
        error(mes);
    }
}


template <typename T>
void
MLPNet<T>::check_w(int l, int n, int npl) const
{
    if ((l < 1) || (l > m_nol)) {
        message mes;
        mes << "invalid layer index: " << l << " (number of layers is"
            << m_nol << ")";
        error(mes);
    }
    if (l == 1) {
        message mes;
        mes << "neurons in layer 1 (input layer) are not connected to any \
other neurons";
        error(mes);
    }
    if ((n < 1) || (n > m_l[l - 1])) {
        message mes;
        mes << "invalid neuron index: " << n << " in layer " << l
            << " (there are " << m_l[l - 1] << " neurons in this layer)";
        error(mes);
    }
    if (npl < 0) error("negative neuron index");
    if (npl) check_n(l - 1, npl);
}


template <typename T>
void
MLPNet<T>::check_w(int i) const
{
    if ((i < 1) || (i > m_w_p[m_nol])) {
        message mes;
        mes << "invalid weight index: " << i << " (total number of weights is "
            << m_w_p[m_nol] << ")";
        error(mes);
    }
}



template <typename T>
T
MLPNet<T>::get_w(int l, int n, int npl) const
{
    check_w(l, n, npl);
    int ind = weight_ind(l, n, npl);
    if (!m_w_fl[ind]) {
        message mes;
        if (npl) {
            mes << "connection between neuron " << n << " in layer " << l
                << " and neuron " << npl << " in layer " << (l - 1)
                << " is off";
        } else {
            mes << "bias of neuron " << n << " in layer " << l
                << " is off";
        }
        error(mes);
    }
    return m_w_val[ind];
}


template <typename T>
T
MLPNet<T>::get_w(int i) const
{
    check_w(i);
    int ind = i - 1;
    if (!m_w_fl[ind]) {
        message mes;
        mes << "weigth " << i << " is off";
        error(mes);
    }
    return m_w_val[ind];
}



template <typename T>
void
MLPNet<T>::set_w(int l, int n, int npl, T w)
{
    check_w(l, n, npl);
    int ind = weight_ind(l, n, npl);
    if (!m_w_fl[ind]) {
        message mes;
        if (npl) {
            mes << "connection between neuron " << n << " in layer " << l
                << " and neuron " << npl << " in layer " << (l - 1)
                << " is off";
        } else {
            mes << "bias of neuron " << n << " in layer " << l
                << " is off";
        }
        error(mes);
    }
    m_w_val[ind] = w;
}


template <typename T>
void
MLPNet<T>::set_w(int i, T w)
{
    check_w(i);
    int ind = i - 1;
    if (!m_w_fl[ind]) {
        message mes;
        mes << "weigth " << i << " is off";
        error(mes);
    }
    m_w_val[ind] = w;
}


template <typename T>
bool
MLPNet<T>::is_active(int l, int n, int npl) const
{
    check_w(l, n, npl);
    return m_w_fl[weight_ind(l, n, npl)];
}


template <typename T>
bool
MLPNet<T>::is_active(int i) const
{
    check_w(i);
    return m_w_fl[i - 1];
}



template <typename T>
void
MLPNet<T>::set_active(int l, int n, int npl, bool on)
{
    check_w(l, n, npl);
    mlp_set_active(&m_l[0], &m_n_p[0], &m_n_prev[0], &m_n_next[0],
                   &m_w_p[0], &m_w_val[0], &m_w_fl[0], &m_w_on,
                   l, n, npl, on);
}



template <typename T>
void
MLPNet<T>::set_active(int i, bool on)
{
    check_w(i);
    mlp_set_active(&m_l[0], &m_n_p[0], &m_n_prev[0], &m_n_next[0],
                   &m_w_p[0], &m_w_val[0], &m_w_fl[0], &m_w_on,
                   i, on);
}



template <typename T>
int
MLPNet<T>::get_abs_w_idx(int i) const
{
    if ((i < 1) || (i > m_w_on)) {
        message mes;
        mes << "invalid 1-based active weight index: " << i
            << ", numer of active weights is " << m_w_on;
        error(mes);
    }

    return mlp_get_abs_w_idx(&m_w_fl[0], i);
}



template <typename T>
void
MLPNet<T>::get_ln_idx(int i, int &l, int &n, int &npl) const
{
    check_w(i);
    mlp_get_lnn_idx(&m_l[0], &m_w_p[0], i, l, n, npl);
}



// ==================================================================
// Weights - randomising, retrieving and setting
// ==================================================================
template <typename T>
void
MLPNet<T>::rnd_weights(T a)
{
    for (int i = 0, n = m_w_p[m_nol]; i < n; ++i)
        if (m_w_fl[i])
            m_w_val[i] = (T)2 * a * ((T) ::rand() / (T) RAND_MAX - (T)0.5);
}


template <typename T>
Matrix<T>
MLPNet<T>::get_weights() const
{
    Matrix<T> ret(m_w_on, 1);
    for (int i = 0, j = 1, n = m_w_p[m_nol]; i < n; ++i)
        if (m_w_fl[i]) ret.elem(j++) = m_w_val[i];

    return ret;
}


template <typename T>
void
MLPNet<T>::set_weights(const Matrix<T> &w, bool mk_zeros_inactive)
{
    if (w.cols() != 1) {
        message mes;
        mes << "weights should be provided as a column vector (input is "
            << w.rows() << "x" << w.cols() << ")";
        error(mes);
    }
    if (m_w_on != w.size()) {
        message mes;
        mes << "no. of active weights (" << m_w_on
            << ") and weights provided (" << w.size() << ") disagree";
        error(mes);
    }

    if (mk_zeros_inactive) {
        for (int i = 0, j = 1, n = m_w_p[m_nol]; i < n; ++i) {
            if (m_w_fl[i]) {
                m_w_val[i] = w.elem(j++);
                if (m_w_val[i] == T()) {
                    --m_w_on;
                    int l, n, npl;
                    get_ln_idx(i + 1, l, n, npl);
                    if (npl) {
                        --m_n_next[neuron_ind(l - 1, npl)];
                        --m_n_prev[neuron_ind(l, n)];
                    }
                }
            }
        }
    } else {
        for (int i = 0, j = 1, n = m_w_p[m_nol]; i < n; ++i)
            if (m_w_fl[i]) m_w_val[i] = w.elem(j++);
    }
}


// ==================================================================
// Save and load
// ==================================================================
template <typename T>
bool
MLPNet<T>::save(const std::string &fname) const
{
    if (!m_nol) error("trying to save uninitialised (empty) network");
    return mlp_save_txt(fname, m_name, m_l, m_w_val, m_w_fl,
                        m_af, m_af_p);

}



template <typename T>
bool
MLPNet<T>::load(const std::string &fname)
{
    clear();
    bool ok = mlp_load_txt(fname, m_name, m_l, m_n_p, m_n_prev, m_n_next,
                           m_w_p, m_w_val, m_w_fl, m_w_on,
                           m_af, m_af_p);
    if (!ok) {
        clear();
    } else {
        m_nol = m_l.size();
    }
    return ok;
}



// ==================================================================
// Export
// ==================================================================
template <typename T>
bool
MLPNet<T>::export_C(const std::string &fname, bool with_bp) const
{
    if (!m_nol) error("trying to export uninitialised (empty) network");
    return mlp_export_C(fname, m_name, m_l, m_n_p, m_w_val, m_w_fl, m_w_on,
                        m_af, m_af_p, with_bp,
                        (T*)0, (T*)0, (T*)0, (T*)0, (T*)0, (T*)0);
}

template <typename T>
bool
MLPNet<T>::export_C(const std::string &fname,
                    const Matrix<T> &A, const Matrix<T> &b,
                    const Matrix<T> &C, const Matrix<T> &d, bool with_bp) const
{
    if (!m_nol) error("trying to export uninitialised (empty) network");
    if ((A.rows() != m_l[0]) || (A.rows() != A.cols()))
        error("invalid input transformation matrix");
    if ((b.rows() != m_l[0]) || (b.cols() != 1))
        error("invalid input translation vector");
    if ((C.rows() != m_l[m_nol - 1]) || (C.rows() != C.cols()))
        error("invalid output transformation matrix");
    if ((d.rows() != m_l[m_nol - 1]) || (d.cols() != 1))
        error("invalid output translation vector");
    if (with_bp) {
        try {
            Matrix<T> E = solve(C);
            Matrix<T> f = -E * d;
            return mlp_export_C(fname, m_name, m_l, m_n_p, m_w_val, m_w_fl, m_w_on, m_af, m_af_p,
                                with_bp, A.ptr(), b.ptr(), C.ptr(), d.ptr(), E.ptr(), f.ptr());
        } catch (exception &e) {
            error("output transformation is not invertible");
        }
    }
    return mlp_export_C(fname, m_name, m_l, m_n_p, m_w_val, m_w_fl, m_w_on, m_af, m_af_p,
                        with_bp, A.ptr(), b.ptr(), C.ptr(), d.ptr(), (T*)0, (T*)0);
}




// ==================================================================
// Feed forward, MSE, backpropagation, gradients
// ==================================================================
template <typename T>
void
MLPNet<T>::check_in(int r, int c) const
{
    if (r < 1) {
        message mes;
        mes << "input data must have at least one row ("
            << r << "x" << c << " matrix provided)";
        error(mes);
    }
    if (c != m_l[0]) {
        message mes;
        mes << "no. of input neurons (" << m_l[0]
            << ") and columns in input matrix ("
            << c << ") disagree";
        error(mes);
    }
}

template <typename T>
void
MLPNet<T>::check_inout(int ri, int ci, int ro, int co) const
{
    if (ri < 1) {
        message mes;
        mes << "input data must have at least one row ("
            << ri << "x" << ci << " matrix provided)";
        error(mes);
    }
    if (ri != ro) {
        message mes;
        mes << "no. of rows in input (" << ri
            << ") and in output matrix ("
            << ro << ") disagree";
        error(mes);
    }
    if (ci != m_l[0]) {
        message mes;
        mes << "no. of input neurons (" << m_l[0]
            << ") and columns in input matrix ("
            << ci << ") disagree";
        error(mes);
    }
    if (co != m_l[m_nol - 1]) {
        message mes;
        mes << "no. of output neurons (" << m_l[m_nol - 1]
            << ") and columns in output matrix ("
            << co << ") disagree";
        error(mes);
    }
}




template <typename T>
Matrix<T>
MLPNet<T>::eval(const Matrix<T> &input) const
{
    check_in(input.rows(), input.cols());

    int r = input.rows();
    Matrix<T> res(input.rows(), m_l[m_nol - 1]);
    fcnn::internal::eval(&m_l[0], m_l.size(), &m_n_p[0],
                         &m_w_val[0], &m_af[0], &m_af_p[0],
                         r, input.ptr(), res.ptr());

    return res;
}






template <typename T>
T
MLPNet<T>::mse(const Matrix<T> &input, const Matrix<T> &output) const
{
    check_inout(input.rows(), input.cols(), output.rows(), output.cols());

    int r = input.rows();
    return fcnn::internal::mse(&m_l[0], m_l.size(), &m_n_p[0],
                               &m_w_val[0], &m_af[0], &m_af_p[0],
                               r, input.ptr(), output.ptr());
}




template <typename T>
std::pair<Matrix<T>, T>
MLPNet<T>::grad(const Matrix<T> &input, const Matrix<T> &output) const
{
    check_inout(input.rows(), input.cols(), output.rows(), output.cols());

    Matrix<T> gradient(m_w_on, 1);
    T se;
    se = fcnn::internal::grad(&m_l[0], m_l.size(), &m_n_p[0],
                              &m_w_p[0], &m_w_fl[0], &m_w_val[0],
                              &m_af[0], &m_af_p[0],
                              input.rows(), input.ptr(), output.ptr(), gradient.ptr());
    return std::pair<Matrix<T>, T>(gradient, se);
}



template <typename T>
Matrix<T>
MLPNet<T>::gradi(const Matrix<T> &input, const Matrix<T> &output, int i) const
{
    check_inout(input.rows(), input.cols(), output.rows(), output.cols());
    if ((i < 1) || (i > input.rows())) {
        message mes;
        mes << "invalid record (row) index " << i << "; data has "
            << input.rows() << " records (rows)";
        error(mes);
    }

    Matrix<T> gradient(m_w_on, 1);
    fcnn::internal::gradi(&m_l[0], m_l.size(), &m_n_p[0],
                          &m_w_p[0], &m_w_fl[0], &m_w_val[0],
                          &m_af[0], &m_af_p[0],
                          input.rows(), i - 1, input.ptr(), output.ptr(), gradient.ptr());
    return gradient;
}




template <typename T>
Matrix<T>
MLPNet<T>::gradij(const Matrix<T> &input, int i) const
{
    check_in(input.rows(), input.cols());
    if ((i < 1) || (i > input.rows())) {
        message mes;
        mes << "invalid record (row) index " << i << "; data has "
            << input.rows() << " records (rows)";
        error(mes);
    }

    Matrix<T> gradients(m_w_on, m_l[m_nol - 1]);
    fcnn::internal::gradij(&m_l[0], m_l.size(), &m_n_p[0],
                           &m_w_p[0], &m_w_fl[0], &m_w_val[0], m_w_on,
                           &m_af[0], &m_af_p[0],
                           input.rows(), i - 1, input.ptr(), gradients.ptr());
    return gradients;
}




template <typename T>
Matrix<T>
MLPNet<T>::jacob(const Matrix<T> &input, int i) const
{
    check_in(input.rows(), input.cols());
    if ((i < 1) || (i > input.rows())) {
        message mes;
        mes << "invalid record (row) index " << i << "; data has "
            << input.rows() << " records (rows)";
        error(mes);
    }

    Matrix<T> jac(m_l[0], m_l[m_nol - 1]);
    fcnn::internal::jacob(&m_l[0], m_l.size(), &m_n_p[0],
                          &m_w_p[0], &m_w_fl[0], &m_w_val[0], m_w_on,
                          &m_af[0], &m_af_p[0],
                          input.rows(), i - 1, input.ptr(), jac.ptr());
    return jac;
}




// ==================================================================
// Instantiations
// ==================================================================
template class MLPNet<float>;
template class MLPNet<double>;

