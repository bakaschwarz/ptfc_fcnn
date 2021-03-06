/* fcnn/fcnncfg.h.in.  Generated from configure.ac by autoheader.  */


/** \file fcnncfg.h
 *  \brief FCNN configuration file.
 */

#ifndef FCNNCFG_H

#define FCNNCFG_H


/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
#undef F77_DUMMY_MAIN

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
#undef F77_FUNC

/* As F77_FUNC, but for C identifiers containing underscores. */
#undef F77_FUNC_

/* Define if F77 and FC dummy `main' functions are identical. */
#undef FC_DUMMY_MAIN_EQ_F77

/* Define if you have a BLAS library. */
#undef HAVE_BLAS

/* Define to 1 if you have the <dlfcn.h> header file. */
#undef HAVE_DLFCN_H

/* Define to 1 if you have the `gettimeofday' function. */
#undef HAVE_GETTIMEOFDAY

/* Define to 1 if you have the <inttypes.h> header file. */
#undef HAVE_INTTYPES_H

/* Define if you have a LAPACK library. */
#undef HAVE_LAPACK

/* Define to 1 if you have the `m' library (-lm). */
#undef HAVE_LIBM

/* Define to 1 if you have the <memory.h> header file. */
#undef HAVE_MEMORY_H

/* Define if OpenMP is supported. */
#undef HAVE_OPENMP

/* Define to 1 if you have the <stdint.h> header file. */
#undef HAVE_STDINT_H

/* Define to 1 if you have the <stdlib.h> header file. */
#undef HAVE_STDLIB_H

/* Define to 1 if you have the <strings.h> header file. */
#undef HAVE_STRINGS_H

/* Define to 1 if you have the <string.h> header file. */
#undef HAVE_STRING_H

/* Define to 1 if you have the <sys/stat.h> header file. */
#undef HAVE_SYS_STAT_H

/* Define to 1 if you have the <sys/time.h> header file. */
#undef HAVE_SYS_TIME_H

/* Define to 1 if you have the <sys/types.h> header file. */
#undef HAVE_SYS_TYPES_H

/* Define to 1 if you have the <unistd.h> header file. */
#undef HAVE_UNISTD_H

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#undef LT_OBJDIR

/* Name of package */
#undef PACKAGE

/* Define to the address where bug reports for this package should be sent. */
#undef PACKAGE_BUGREPORT

/* Define to the full name of this package. */
#undef PACKAGE_NAME

/* Define to the full name and version of this package. */
#undef PACKAGE_STRING

/* Define to the one symbol short name of this package. */
#undef PACKAGE_TARNAME

/* Define to the home page for this package. */
#undef PACKAGE_URL

/* Define to the version of this package. */
#undef PACKAGE_VERSION

/* Define to 1 if you have the ANSI C header files. */
#undef STDC_HEADERS

/* Version number of package */
#define VERSION "0.5.1"


#ifdef R_SHAREDLIB
#include "fcnncfg.R.h" /* distributed with R package only */
#endif /* R_SHAREDLIB */

#endif /* FCNNCFG_H */
