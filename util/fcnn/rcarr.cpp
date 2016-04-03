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

/** \file rcarr.cpp
 *  \brief Reference counted array.
 */


#include <rcarr.h>
#include <error.h>
#include <utils.h>
#include <cstdlib>
#include <cstring>



using namespace std;
using namespace fcnn;
using namespace fcnn::internal;


// Constructor
template <typename T>
rcarr<T>::rcarr(int n)
{
    alloc(n);
}


// Constructor
template <typename T>
rcarr<T>::rcarr(T *ptr, int n)
{
    alloc(n);
    read_from(ptr);
}


// Copy constructor
template <typename T>
rcarr<T>::rcarr(const rcarr<T> &a)
{
    m_ptr = a.m_ptr;
    m_rc = a.m_rc;
    m_size = a.m_size;

    if (m_rc) (*m_rc)++;
}


// Reset function
template <typename T>
rcarr<T>&
rcarr<T>::reset(int n)
{
    dec_rc();
    alloc(n);

    return *this;
}



// Destructor
template <typename T>
rcarr<T>::~rcarr()
{
    dec_rc();
}


// Assignment (shallow)
template <typename T>
rcarr<T>&
rcarr<T>::operator=(const rcarr<T> &a)
{
    if (m_ptr == a.m_ptr) return *this;

    dec_rc();

    m_ptr = a.m_ptr;
    m_rc = a.m_rc;
    m_size = a.m_size;

    if (m_rc) (*m_rc)++;

    return *this;
}


// Creating copies
template <typename T>
rcarr<T>
rcarr<T>::copy() const
{
    rcarr<T> tmp(m_size);

    if (m_size) tmp.read_from(m_ptr + 1);

    return tmp;
}


// Make this data unique
template <typename T>
void
rcarr<T>::mkunique()
{
    if ((m_rc) && (*m_rc > 1))
        *this = copy();
}


// Setting all entries to given value
template <typename T>
void
rcarr<T>::set_all_to(const T &val)
{
    if (val == T()) {
        memset(m_ptr + 1, 0, m_size * sizeof(T));
        return;
    }
    for (int i = 1; i <= m_size; i++) m_ptr[i] = val;
}




// Memory allocation
template <typename T>
void
rcarr<T>::alloc(int n)
{
    if (n < 0) error("negative array size");
    if (n)
    {
        m_ptr = (T*) malloc(sizeof(T) * n);
        if (!m_ptr) {
            message mes;
            mes << "failed to allocate " << ((int) sizeof(T) * n)
                << "B of memory";
            error(mes);
        }
        m_ptr--;
        m_size = n;
        m_rc = (int*) malloc(sizeof(int));
        *m_rc = 1;
    }
    else
    {
        m_size = 0;
        m_ptr = 0;
        m_rc = 0;
    }
}


// Decreasing reference count
template <typename T>
void
rcarr<T>::dec_rc()
{
    if (m_rc)
    {
        --(*m_rc);

        if ((*m_rc) == 0)
        {
            free(m_rc);
            free(++m_ptr);
            m_rc = 0;
        }
    }
}


// Copying data from pointer
template <typename T>
void
rcarr<T>::read_from(const T *ptr)
{
    memcpy((void*) (m_ptr + 1), (const void*) (ptr), m_size * sizeof(T));
}





// Instantiations
// ===================================================================
template class rcarr<double>;
template class rcarr<float>;

