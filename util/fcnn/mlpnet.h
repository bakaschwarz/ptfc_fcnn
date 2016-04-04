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

/** \file mlpnet.h
 *  \brief Multilayer perceptron network.
 */


#ifndef FCNN_MPLNET_H

#define FCNN_MPLNET_H


#include <string>
#include <vector>
#include <map>
#include <mat.h>
#include <dataset.h>
#include <activation.h>


namespace fcnn {


template <typename T> class MLPNet;

/// Merge two networks (they must have the same number of layers).
template <typename T> MLPNet<T> merge(const MLPNet<T> &A, const MLPNet<T> &B, bool same_inputs = false);
/// Connect one network outputs to another network inputs (the numbers of output
/// and input neurons must agree).
template <typename T> MLPNet<T> stack(const MLPNet<T> &A, const MLPNet<T> &B);



/// Multilayer perceptron network implementation.
template <typename T>
class MLPNet {
  public:
    /// Default constructor (uninitialised network).
    MLPNet() : m_nol(0) { ; }
    /// Constructor. All weights are set to 0, but active.
    MLPNet(const std::vector<int> &layers);
    /// Constructor. All weights are set to 0, but active.
    MLPNet(int no_of_layers, ...);
    /// Destructor
    ~MLPNet() { ; }

    /// Construct new network. All weights are set to 0, but active.
    void construct(const std::vector<int> &layers);
    /// Construct new network. Active weights are set based on the 3rd argument.
    void construct(const std::vector<int> &layers,
                   const std::vector<int> &w_active,
                   const std::vector<T> &w_vals);
    /// Reconstruct network by adding and/or reordeing input neurons. All new
    /// connections are inactive. Requires the new total no. of input neurons
    /// and a map of old inputs to new ones (1-based indexing).
    void expand_reorder_inputs(int newnoinp, const std::map<int, int> &m);

    /// Remove inactive (i.e. not connected to previous or next layer) neurons
    /// in the hidden layers. Returns no. of neurons and associated weights removed.
    std::pair<int, int> rm_neurons(bool report = false);
    /// Remove inactive (i.e. not connected to the next layer) input neurons.
    /// Returns (1-based) indices of neurons that where not removed.
    std::vector<int> rm_input_neurons(bool report = false);

    /// Load network from file, returns true on success.
    bool load(const std::string &fname);
    /// Save network to file, returns true on success.
    bool save(const std::string &fname) const;

    /// Export network to a C function, returns true on success.
    bool export_C(const std::string &fname, bool with_bp = false) const;
    /// Export network to a C function with input and output
    /// affine transformations of the form \f$Ax+b\f$ (input)
    /// and \f$Cx+d\f$ (output); returns true on success. When with_wp
    /// is set to true, backpropagation code (for online learning)
    /// is also exported.
    bool export_C(const std::string &fname,
                  const Matrix<T> &A, const Matrix<T> &b,
                  const Matrix<T> &C, const Matrix<T> &d,
                  bool with_bp = false) const;

    /// Is network initialised?
    operator bool() { return m_nol; }

    /// Set activation function (and its parameter) for layer l. When parameter
    /// is 0, the default parameter value for given activation is set.
    void set_act_f(int l, int af, T param = 0);

    /// Set network name.
    void set_name(const std::string &name) { m_name = name; }
    /// Get network name.
    std::string get_name() const { return m_name; }

    /// Get no. of layers
    inline int no_layers() const { return m_nol; }
    /// Get no. of neurons in given layer
    int no_neurons(int l) const;
    /// Return total no. of weights (connections and biases) (including
    /// inactive).
    inline int total_w() const { return m_w_p[m_nol]; }
    /// Return no. of non-zero weights (active connections and biases).
    inline int active_w() const { return m_w_on; }

    /// Get weight of connection between neuron n in layer l and neuron
    /// npl in layer l-1. Npl set to zero denotes bias of neuron n in layer l.
    T get_w(int l, int n, int npl) const;
    /// Get value of weight with given (1-based) index.
    T get_w(int i) const;
    /// Set weight of connection between neuron n in layer l and neuron
    /// npl in layer l-1.
    void set_w(int l, int n, int npl, T w);
    /// Set value of weight with given (1-based) index.
    void set_w(int i, T w);
    /// Is connection between neuron n in layer l and neuron npl in layer l-1
    /// active? Npl set to zero denotes bias of neuron n in layer l.
    bool is_active(int l, int n, int npl) const;
    /// Is weight with given (1-based) index active?
    bool is_active(int i) const;
    /// Set connection between neuron n in layer l and neuron npl in layer l-1
    /// on/off. Npl set to zero denotes bias of neuron n in layer l.
    void set_active(int l, int n, int npl, bool on = true);
    /// Set weight with given (1-based) index on/off.
    void set_active(int i, bool on = true);

    /// Get absolute (i.e. including inactive weights) 1-based index of a weight
    /// given its (1-based) index within active ones.
    int get_abs_w_idx(int i) const;
    /// Given 1-based weight absolute index (including those turned off)
    /// set layer, neuron and previous neuron 1-based neuron indices
    /// (0 for npl means bias).
    void get_ln_idx(int i, int &l, int &n, int &npl) const;

    /// Draw random initial weights from \f$(-a, a)\f$.
    void rnd_weights(T a = (T)0.2);
    /// Get vector of weights of active connections as column vector.
    Matrix<T> get_weights() const;
    /// Set vector of weights of active connections. Admits column vector.
    /// If mk_zeros_inactive is true, sets the weights corresponding
    /// to zeros off.
    void set_weights(const Matrix<T>&, bool mk_zeros_inactive = false);

    /// Evaluate output given input.
    Matrix<T> eval(const Matrix<T> &input) const;
    /// Compute MSE for \f$N\f$ data records and \f$O\f$ outputs given by
    /// \f$\frac{1}{2 N O} \sum_{n=1}^N \sum_{o=1}^O {e_o^n}^2\f$.
    T mse(const Matrix<T> &input, const Matrix<T> &output) const;
    /// Compute MSE for \f$N\f$ data records and \f$O\f$ outputs given by
    /// \f$\frac{1}{2 N O} \sum_{n=1}^N \sum_{o=1}^O {e_o^n}^2\f$.
    T mse(const Dataset<T> &dat) const
    {
        return mse(dat.get_input(), dat.get_output());
    }

    /// Compute gradient (column vector) of MSE (derivatives w.r.t. active weights)
    /// given input and expected output. Returns MSE as second element
    /// in the pair. This function is useful when implementing batch teaching
    /// algorithms.
    std::pair<Matrix<T>, T> grad(const Matrix<T> &input,
                                 const Matrix<T> &output) const;
    /// Compute gradient (column vector) of MSE (derivatives w.r.t. active weights)
    /// given input and expected output. Returns MSE as second element
    /// in the pair. This function is useful when implementing batch teaching
    /// algorithms.
    std::pair<Matrix<T>, T> grad(const Dataset<T> &dat) const
    {
        return grad(dat.get_input(), dat.get_output());
    }
    /// Compute gradient (column vector) of MSE (derivatives w.r.t. active weights)
    /// given input and expected output using ith row of data only. This is
    /// normalised by the number of outputs only, the average over all rows
    /// (all i) returns the same as grad(input, output). This function is useful
    /// when implementing on-line teaching algorithms.
    Matrix<T> gradi(const Matrix<T> &input, const Matrix<T> &output,
                    int i) const;
    /// Compute gradient (column vector) of MSE (derivatives w.r.t. active weights)
    /// given input and expected output using ith row of data only. This is
    /// normalised by the number of outputs only, the average over all rows
    /// (all i) returns the same as grad(input, output). This function is useful
    /// when implementing on-line teaching algorithms.
    Matrix<T> gradi(const Dataset<T> &dat, int i) const
    {
        return gradi(dat.get_input(), dat.get_output(), i);
    }
    /// Compute gradients of networks outputs, i.e the derivatives of outputs
    /// w.r.t. active weights, at given data row. The derivatives of outputs
    /// are placed in subsequent columns of the returned matrix. Scaled by
    /// the output errors and averaged they give the same as gradi(input, output, i).
    Matrix<T> gradij(const Matrix<T> &input, int i) const;
    /// Compute gradients of networks outputs, i.e the derivatives of outputs
    /// w.r.t. active weights, at given data row. The derivatives of outputs
    /// are placed in subsequent columns of the returned matrix. Scaled by
    /// the output errors and averaged they give the same as gradi(input, output, i).
    Matrix<T> gradij(const Dataset<T> &dat, int i) const
    {
        return gradij(dat.get_input(), i);
    }
    /// Compute the Jacobian of network transformation, i.e the derivatives
    /// of outputs w.r.t. network inputs, at given data row. The derivatives
    /// of outputs are placed in subsequent columns of the returned matrix.
    Matrix<T> jacob(const Matrix<T> &input, int i) const;
    /// Compute the Jacobian of network transformation, i.e the derivatives
    /// of outputs w.r.t. network inputs, at given data row. The derivatives
    /// of outputs are placed in subsequent columns of the returned matrix.
    Matrix<T> jacob(const Dataset<T> &dat, int i) const
    {
        return jacob(dat.get_input(), i);
    }

#ifdef FCNN_DEBUG
    /// Dump on std::cerr (for debugging purposes)
    void dump() const;
#endif /* FCNN_DEBUG */

  private:
    /// Network name,
    std::string m_name;
    /// No. of layers.
    int m_nol;
    /// Layers.
    std::vector<int> m_l;
    /// Neuron 'pointers'.
    std::vector<int> m_n_p;
    /// No. of connected neurons in the previous layer.
    std::vector<int> m_n_prev;
    /// No. of connected neurons in the next layer.
    std::vector<int> m_n_next;
    /// Weight 'pointers'.
    std::vector<int> m_w_p;
    /// Weight values.
    std::vector<T> m_w_val;
    /// Weights on/off.
    std::vector<int> m_w_fl;
    /// No. of weights on.
    int m_w_on;
    /// Activation functions.
    std::vector<int> m_af;
    /// Activation functions' parameters.
    std::vector<T> m_af_p;

    /// Clear existing structure.
    void clear();

    /// Check layer index.
    void check_l(int l) const;
    /// Check neuron index.
    void check_n(int l, int n) const;
    /// Check weight index.
    void check_w(int l, int n, int npl) const;
    /// Check weight index.
    void check_w(int i) const;
    /// Return index of nth neuron in layer l.
    inline int neuron_ind(int l, int n) const {
        return m_n_p[l - 1] + n - 1;
    }
    /// Return weight index for connection between neuron n in layer l and
    /// neuron npl in previous layer.
    inline int weight_ind(int l, int n, int npl) const {
        return m_w_p[l - 1] + (n - 1) * (m_l[l - 2] + 1) + npl;
    }

    /// Check input data (matrix size).
    void check_in(int r, int c) const;
    /// Check input and output data (matrix).
    void check_inout(int ri, int ci, int ro, int co) const;

    friend MLPNet<T> merge<>(const MLPNet<T>&, const MLPNet<T>&, bool);
    friend MLPNet<T> stack<>(const MLPNet<T>&, const MLPNet<T>&);

}; /* class template MLPNet */


} /* namespace fcnn */


#endif /* FCNN_MPLNET_H */
