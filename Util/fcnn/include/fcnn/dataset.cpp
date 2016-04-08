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

/** \file dataset.cpp
 *  \brief Dataset.
 */


#include <fcnn/dataset.h>
#include <fcnn/utils.h>
#include <fcnn/error.h>
#include <iomanip>


using namespace fcnn;
using namespace fcnn::internal;


//namespace fcnn {

template <typename T>
void
Dataset<T>::set(const Matrix<T> &in, const Matrix<T> &out,
                const std::string &descr,
                const std::vector<std::string> &record_descr)
{
    int ri = in.rows(), ro = out.rows(), ci = in.cols(), co = out.cols();

    if (ri != ro) {
        message mes;
        mes << "no. of rows in input (" << ri
        << ") and in output matrix ("
        << ro << ") disagree";
        error(mes);
    }

    if (!ci) {
        error("empty input");
    }

    if (!co) {
        error("empty output");
    }

    int rd = record_descr.size();
    if (rd) {
        if (ri != rd) {
            message mes;
            mes << "no. of records (" << ri
                << ") and record descriptions ("
                << rd << ") disagree";
            error(mes);
        }
        m_rec_info = record_descr;
    } else {
        m_rec_info.assign(ri, "");
    }

    m_in = in;
    m_out = out;
    m_info = descr;
}



template <typename T>
std::string
Dataset<T>::get_record_info(int i) const
{
    int r = m_rec_info.size();
    if ((i < 1) || (i > r)) {
        message mes;
        mes << "invalid record index: index " << i << ", no. of records "
            << "in dataset: " << r;
        error(mes);
    }

    return m_rec_info[i - 1];
}



template <typename T>
void
Dataset<T>::set_record_info(int i, const std::string &info)
{
    int r = m_rec_info.size();
    if ((i < 1) || (i > r)) {
        message mes;
        mes << "invalid record index: index " << i << ", no. of records "
            << "in dataset: " << r;
        error(mes);
    }

    m_rec_info[i - 1] = info;
}




template <typename T>
bool
Dataset<T>::save(const std::string &fname, bool write_info) const
{
    std::ofstream os;
    os.open(fname.c_str());
    if (os.fail()) return false;
    int i, j, r = m_in.rows(), ci = m_in.cols(), co = m_out.cols();
    if (write_info) {
        std::string info = m_info;
        if (info == "") info = "untitled dataset";
        if (!write_comment(os, info)) return false;
    }

    os << r << ' ' << ci << ' ' << co << '\n';

    os << std::setprecision(precision<T>::val);
    for (i = 1; i <= r; ++i) {
        if (write_info) {
            std::string info = m_rec_info[i - 1];
            if (info == "") info = num2str(i);
            if (!write_comment(os, info)) return false;
        }
        for (j = 1; j < ci; ++j) os << m_in.elem(i, j) << ' ';
        os << m_in.elem(i, j) << '\n';
        if (os.fail()) return false;
        for (j = 1; j < co; ++j) os << m_out.elem(i, j) << ' ';
        os << m_out.elem(i, j) << '\n';
        if (os.fail()) return false;
    }
    os << '\n';
    os.close();
    if (os.fail()) return false;
    return true;
}




template <typename T>
bool
Dataset<T>::load(const std::string &fname, bool read_info)
{
    m_info.clear();
    m_rec_info.clear();
    m_in.reset();
    m_out.reset();

    std::ifstream is;
    is.open(fname.c_str());
    if (is.fail()) return false;

    int r, ci, co;
    std::string cm;

    skip_blank(is);
    if (read_info) { read_comment(is, cm); } else skip_comment(is);
    if (is.fail()) return false;
    if (is_eol(is)) return false;
    if (!read<int>(is, r)) return false;
    if (is_eol(is)) return false;
    if (!read<int>(is, ci)) return false;
    if (is_eol(is)) return false;
    if (!read<int>(is, co)) return false;
    if (!is_eol(is)) return false;
    skip_blank(is);
    if (is.fail()) return false;
    if ((r < 1) || (ci < 1) || (co < 1)) return false;

    m_info = cm;
    m_in.reset(r, ci);
    m_out.reset(r, co);
    m_rec_info.assign(r, "");

    for  (int i = 1; i <= r; ++i) {
        T d;
        if (read_info) {
            if (read_comment(is, cm)) m_rec_info[i - 1] = cm;
        } else skip_comment(is);
        for (int j = 1; j <= ci; ++j) {
            if (is_eol(is)) goto err;
            if (!read<T>(is, d)) goto err;
            m_in(i, j) = d;
        }
        if (!is_eol(is)) goto err;
        for (int j = 1; j <= co; ++j) {
            if (is_eol(is)) goto err;
            if (!read<T>(is, d)) goto err;
            m_out(i, j) = d;
        }
        if (i < r) {
            if (!is_eol(is)) goto err;
        } else {
            skip_all(is);
            if (!is.eof()) goto err;
        }
    }
    return true;

err:
    m_info.clear();
    m_rec_info.clear();
    m_in.reset();
    m_out.reset();
    return false;
}

// Instantiations
template class fcnn::Dataset<double>;
template class fcnn::Dataset<float>;
