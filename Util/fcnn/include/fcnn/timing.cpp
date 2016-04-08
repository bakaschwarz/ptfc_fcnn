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

/** \file timing.cpp
 *  \brief Tic & toc.
 */


#include <fcnn/timing.h>
#include <fcnn/fcnncfg.h>



#if defined(HAVE_GETTIMEOFDAY)
#include <sys/time.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <ctime>
#endif




namespace
{

    bool tic_started = false;
#if defined(HAVE_GETTIMEOFDAY)
    struct timeval tic_timeval;
#elif defined(_WIN32)
    __int64 tic_int64;
#else
    clock_t tic_clock_t;
#endif

} /* namespace */



void
fcnn::tic(void)
{
    tic_started = true;
#if defined(HAVE_GETTIMEOFDAY)
    gettimeofday(&tic_timeval, 0);
#elif defined(_WIN32)
    QueryPerformanceCounter((LARGE_INTEGER*) &tic_int64);
#else
    tic_clock_t = clock();
#endif
}


double
fcnn::toc(void)
{
    double t;
#if defined(HAVE_GETTIMEOFDAY)
    struct timeval toc_timeval;
#elif defined(_WIN32)
    __int64 toc_int64, fr_int64;
#else
    clock_t toc_clock_t;
#endif

    if (tic_started)
    {
        tic_started = false;
#if defined(HAVE_GETTIMEOFDAY)
        gettimeofday(&toc_timeval, 0);
        t = (double) (toc_timeval.tv_sec - tic_timeval.tv_sec)
            + (double) (toc_timeval.tv_usec - tic_timeval.tv_usec) * 1e-6;
#elif defined(_WIN32)
        QueryPerformanceCounter((LARGE_INTEGER*) &toc_int64);
        QueryPerformanceFrequency((LARGE_INTEGER*) &fr_int64);
        t = (double) (toc_int64 - tic_int64) / (double) fr_int64;
#else
        toc_clock_t = clock();
        t = (double) (toc_clock_t - tic_clock_t) / (double) CLOCKS_PER_SEC;
#endif
        return t;
    }

    return 0.;
}



