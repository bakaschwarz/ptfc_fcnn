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

/** \file rcarr.h
 *  \brief Reference counted array.
 */


#ifndef FCNN_RCARR_H

#define FCNN_RCARR_H



namespace fcnn {
namespace internal {


/// This is the base container class for Matrix class. It handles
/// memory allocation, reference counting and provides inlined element access
/// with 1-based indexing.
template <typename T>
class rcarr {
 public:
    /// Constructor (allocates memory).
    explicit rcarr(int = 0);
    /// Constructor from array.
    rcarr(T*, int);
    /// Copy constructor.
    rcarr(const rcarr<T>&);

    /// Reset member function allocates new memory.
    rcarr<T>& reset(int = 0);

    /// Destructor.
    ~rcarr();

    /// Make this data unique.
    void mkunique();

    /// Returns pointer to data.
    inline T* ptr() { return m_ptr + 1; }
    /// Returns pointer to data (const version).
    inline T const* ptr() const { return m_ptr + 1; }

    /// Element access, no index checking.
    inline T& operator[](int i) { return m_ptr[i]; }
    /// Element access (const version), no index checking.
    inline T const& operator[](int i) const { return m_ptr[i]; }

    /// Returns size (number of elements).
    inline int size() const { return m_size; }

    /// Assignment.
    rcarr<T>& operator=(const rcarr<T>&);

    /// Return verbatim copy of data (new memory is allocated).
    rcarr<T> copy() const;

    /// Read (copy) data from pointer.
    void read_from(const T*);

    /// Set all entries to given value.
    void set_all_to(const T&);

 private:
    /// Pointer to data.
    T *m_ptr;
    /// Pointer to the reference count.
    int *m_rc;
    /// Size.
    int m_size;

    /// Memory allocation.
    void alloc(int);
    /// Decrement refcount (called by destructor).
    void dec_rc();

}; /* class template rcarr */



} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_RCARR_H */
