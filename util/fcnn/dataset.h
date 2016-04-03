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

/** \file dataset.h
 *  \brief Dataset.
 */

#ifndef FCNN_DATASET_H

#define FCNN_DATASET_H


#include <mat.h>
#include <string>
#include <vector>
#include <fstream>


namespace fcnn {


/// Class for storing teaching and testing data.
template <typename T>
class Dataset
{
  public:
    /// Default constructor
    Dataset() { ; }

    /// Set new matrices and optionally descriptions (throws on error).
    void set(const Matrix<T> &in, const Matrix<T> &out,
             const std::string &descr = "",
             const std::vector<std::string> &record_descr =
             std::vector<std::string>());

    /// Load data from file, returns true on success.
    bool load(const std::string &fname, bool read_info = true);
    /// Save data to file, returns true on success.
    bool save(const std::string &fname, bool write_info = true) const;

    /// Retrieve input matrix.
    const Matrix<T>& get_input() const { return m_in; }
    /// Retrieve output matrix.
    const Matrix<T>& get_output() const { return m_out; }

    /// Get no. of records.
    inline int no_records() const { return m_in.rows(); }
    /// Get no. of inputs.
    inline int no_inputs() const { return m_in.cols(); }
    /// Get no. of outputs.
    inline int no_outputs() const { return m_out.cols(); }

    /// Retrieve data information.
    inline std::string get_info() const { return m_info; }
    /// Set data information.
    inline void set_info(const std::string &info) { m_info = info; }
    /// Get record information.
    std::string get_record_info(int i) const;
    /// Set record information.
    void set_record_info(int i, const std::string &info);

  private:
    /// Data.
    Matrix<T> m_in, m_out;
    /// Dataset description.
    std::string m_info;
    /// Record descriptions.
    std::vector<std::string> m_rec_info;

}; /* Dataset class template */


} /* namespace fcnn */

#endif /* FCNN_DATASET_H */
