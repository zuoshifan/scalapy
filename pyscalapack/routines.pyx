
cimport numpy as np
import numpy as np


from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free

from scalapack cimport *

from pyscalapack.core cimport *
#from scarray import *
from pyscalapack.blockcyclic import numrc as nrc

cdef int _ONE = 1
cdef int _ZERO = 0


def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def assert_square(A):
    Alist = flatten([A])
    for A in Alist:
        if A.Nr != A.Nc:
            raise Exception("Matrix must be square (has dimensions %i x %i)." % (A.Nr, A.Nc))

def assert_type(A, dtype):
    Alist = flatten([A])
    for A in Alist:
        if A.dtype != dtype:
            raise Exception("Expected Matrix to be of type %s, got %s." % (repr(dtype), repr(A.dtype)))
    


def pdsyevd(mat, destroy = True, upper = True):
    r"""Compute the eigen-decomposition of a symmetric matrix.

    Use Scalapack to compute the eigenvalues and eigenvectors of a
    distributed matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to decompose.
    destroy : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    upper : boolean, optional
        Scalapack uses only half of the matrix, by default the upper
        triangle will be used. Set to False to use the lower triangle.

    Returns
    -------
    evals : np.ndarray
        The eigenvalues of the matrix, they are returned as a global
        numpy array of all values.
    evecs : DistributedMatrix
        The eigenvectors as a DistributedMatrix.
    """
    cdef int lwork, liwork
    cdef double * work
    cdef double wl
    cdef int * iwork
    cdef DistributedMatrix A, evecs

    cdef int info
    cdef np.ndarray evals

    ## Check input
    assert_type(mat, np.float64)
    assert_square(mat)
    
    A = mat if destroy else mat.copy()

    evecs = DistributedMatrix.empty_like(A)
    evals = np.empty(A.Nr, dtype=np.float64)

    liwork = 7*A.Nr + 8*A.context.num_cols + 2
    iwork = <int *>malloc(sizeof(int) * liwork)

    uplo = "U" if upper else "L"

    ## Workspace size inquiry
    lwork = -1
    pdsyevd_("V", uplo, &(A.Nr),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             <double *>np_data(evals),
             <double *>evecs._data(), &_ONE, &_ONE, evecs._getdesc(),
             &wl, &lwork, iwork, &liwork,
             &info);
    
    ## Initialise workspace to correct length
    lwork = <int>wl
    work = <double *>malloc(sizeof(double) * lwork)

    ## Compute eigen problem
    pdsyevd_("V", uplo, &(A.Nr),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             <double *>np_data(evals),
             <double *>evecs._data(), &_ONE, &_ONE, evecs._getdesc(),
             work, &lwork, iwork, &liwork,
             &info);

    free(iwork)
    free(work)

    return (evals, evecs)
    




def pdgemm(DistributedMatrix A, DistributedMatrix B, DistributedMatrix C = None, alpha = 1.0, beta = 1.0, transa = False, transb = False, destroyc = True):
    r"""Compute the eigen-decomposition of a symmetric matrix.

    Use Scalapack to compute the eigenvalues and eigenvectors of a
    distributed matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to decompose.
    destroy : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    upper : boolean, optional
        Scalapack uses only half of the matrix, by default the upper
        triangle will be used. Set to False to use the lower triangle.

    Returns
    -------
    evals : np.ndarray
        The eigenvalues of the matrix, they are returned as a global
        numpy array of all values.
    evecs : DistributedMatrix
        The eigenvectors as a DistributedMatrix.
    """
    cdef int m, n, k
    cdef DistributedMatrix Cm
    cdef int info
    cdef double a, b

    assert_type([A, B, C], np.float64)
    
    a = alpha
    b = beta

    m = A.Nr if not transa else A.Nc
    k = A.Nc if not transa else A.Nr

    n = B.Nc if not transb else B.Nr
    k2 = B.Nr if not transb else B.Nc

    ## Check matrix sizes A, B, are compatible
    if k != k2:
        raise Exception("Matrices A and B have incompatible shapes for multiplication.")

    ## Ensure C has correct size, and copy if required or create if not passed in.
    if C != None:
        if m != C.Nr or n != C.Nc:
            raise Exception("Matrix C is not compatible with matrices A and B.")

        Cm = C if destroyc else C.copy()

    else:
        Cm = DistributedMatrix(globalsize = [m, n], dtype=A.dtype, blocksize = [A.Br, A.Bc], context = A._context)

    tA = "N" if not transa else "T"
    tB = "N" if not transb else "T"

    pdgemm_(tA, tB, &m, &n, &k, &a,
            <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
            <double *>B._data(), &_ONE, &_ONE, B._getdesc(),
            &b,
            <double *>Cm._data(), &_ONE, &_ONE, Cm._getdesc())

    return Cm



def pdpotrf(mat, destroy = True, upper = True):
    r"""Compute the Cholesky decomposition of a symmetric positive definite
    matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to decompose.
    destroy : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    upper : boolean, optional
        Scalapack uses only half of the matrix, by default the upper
        triangle will be used. Set to False to use the lower triangle.

    Returns
    -------
    cholesky : DistributedMatrix
        The Cholesky decomposition of the matrix.
    """
    cdef DistributedMatrix A

    cdef int info


    ## Check input
    assert_type(mat, np.float64)
    assert_square(mat)
    
    A = mat if destroy else mat.copy()

    uplo = "U" if upper else "L"
    
    pdpotrf_(uplo, &(A.Nr),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             &info)

    if info < 0:
        raise Exception("Something weird has happened.")
    elif info > 0:
        raise Exception("Matrix is not positive definite.")

    ## Zero other triangle
    # by default scalapack doesn't touch the other triangle
    # (determined by upper arg). We explicitly zero it here.
    ri, ci = A.indices()
    if upper:
        A.local_array[np.where(ci - ri < 0)] = 0.0
    else:
        A.local_array[np.where(ci - ri > 0)] = 0.0
        
    return A

def pdpotrs(mat, rhs, destroy = True, upper = True):
    r"""Solve the linear system(s) of equations mat*X=rhs using
    Cholesky factorization and Scalapack.
    
    Parameters
    ----------
    mat : DistributedMatrix
        The triangular Cholesky factored matrix as computed by pdpotrf.
    rhs : DistributedMatrix
        Contains one column for each system of equations to solve.
    destroy : boolean, optional
        By default the input rhs is destroyed, if set to False a
        copy is taken and operated on.
    upper : boolean, optional
        Scalapack uses only half of the Cholesky matrix, by default the upper
        triangle will be used. Set to False to use the lower triangle.

    Returns
    -------
    solution : DistributedMatrix
        The solution to mat*X=rhs.
    """
    cdef DistributedMatrix A
    cdef DistributedMatrix B

    cdef int info


    ## Check input
    assert_type(mat, np.float64)
    assert_square(mat)
    if mat.Nr != rhs.Nr:
        msg = "Colesky matrix and solution vector/matrix must have same number "
        msg += "of rows. A has %i and B has %i rows currently." % (mat.Nr,rhs.Nr)
        raise Exception(msg)

    A = mat
    B = rhs if destroy else rhs.copy()

    uplo = "U" if upper else "L"

    pdpotrs_(uplo, &(A.Nr), &(B.Nc),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             <double *>B._data(), &_ONE, &_ONE, B._getdesc(),
             &info)

    if info < 0:
        raise Exception("Something weird has happened.")
    elif info > 0:
        raise Exception("Matrix is not positive definite.")

    # After computation, B becomes X so return that.
    return B

'''
def pdgetrf(mat, destroy = True):
    r"""Compute an LU factorization of a general matrix, using partial
    pivoting with row interchanges.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to decompose.
    destroy : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.

    Returns
    -------
    lu_decomp : DistributedMatrix
        The LU decomposition of the matrix.
    pivots : int array
        An array of length ( LOCr(M_mat)+MB_mat )
        This array contains the pivoting information.
        pivots(i) -> The global row local row i was swapped with.
        This array is tied to the distributed matrix mat.
        http://www.netlib.org/scalapack/double/pdgetrf.f
    """

    cdef DistributedMatrix A

    cdef int * ipiv
    cdef int info
    
    ## Check input
    assert_type(mat, np.float64)
    assert_square(mat)
 
    A = mat if destroy else mat.copy()

    # For pivoting information.
    nr = nrc(A.Nr, A.Br, A.context.row, 0, A.context.num_rows)
    ipiv = <int *>malloc(sizeof(int) * (nr+A.Br))

    pdgetrf_(&(A.Nr), &(A.Nc),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             ipiv, &info)

    if info < 0:
        raise Exception("Something weird has happened.")
    elif info > 0:
        raise Exception("Solving with this decomposition will cause division by 0.")

    return A,ipiv


def pdgetrs(mat, pivots, rhs, destroy = True, trans = False):
    r"""Solve the linear system(s) of equations mat*X=rhs using
    LU decomposition and Scalapack.

    Parameters
    ----------
    mat : DistributedMatrix
        The LU decomposed matrix as computed by pdgetrf.
    pivots : int array
        Contains pivoting information from the LU decomposition.
        See pdgetrf.
    rhs : DistributedMatrix
        Contains one column for each system of equations to solve.
    destroy : boolean, optional
        By default the input rhs is destroyed, if set to False a
        copy is taken and operated on.
    trans : boolean, optional
        Solves mat*X=rhs if False. Solves transpose(mat)*X=rhs if True.

    Returns
    -------
    solution : DistributedMatrix
        The solution to mat*X=rhs.
    """


    cdef DistributedMatrix A
    cdef DistributedMatrix B

    cdef int info

    ## Check input
    assert_type(mat, np.float64)
    assert_square(mat)
    if mat.Nr != rhs.Nr:
        msg = "LU matrix and solution vector/matrix must have same number "
        msg += "of rows. A has %i and B has %i rows currently." % (mat.Nr,rhs.Nr)
        raise Exception(msg)

    A = mat
    B = rhs if destroy else rhs.copy()

    torno = "T" if trans else "N"

    pdgetrs_(torno, &(A.Nr), &(B.Nc),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(), <int *>pivots,
             <double *>B._data(), &_ONE, &_ONE, B._getdesc(),
             &info)

    if info < 0:
        raise Exception("Something weird has happened.")
    elif info > 0:
        raise Exception("This should never occur.")

    return B
'''


def pdgetrf_pdgetrs(mat, rhs, destroy_mat=True, destroy_rhs=True, trans=False):
    r"""Solve the linear system(s) of equations mat*X=rhs using
    LU decomposition and Scalapack.
    There is pivoting information as a C array that cannot be returned from
    the factorization that the solving function needs, this is all in one.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to decompose.
    rhs : DistributedMatrix
        Contains one column for each system of equations to solve.
    destroy_mat : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    destroy_rhs : boolean, optional
        By default the input rhs is destroyed, if set to False a
        copy is taken and operated on.
    trans : boolean, optional
        Solves mat*X=rhs if False. Solves transpose(mat)*X=rhs if True.

    Returns
    -------
    lu_decomp : DistributedMatrix
        The LU decomposition of mat in one matrix.
        Diagonal elements of L are not stored.
    solution : DistributedMatrix
    The solution to mat*X=rhs.
"""

    cdef DistributedMatrix A
    cdef DistributedMatrix B

    cdef int * ipiv
    cdef int info

    ## Check input
    assert_type(mat, np.float64)
    assert_square(mat)
    if mat.Nr != rhs.Nr:
        msg = "LU matrix and solution vector/matrix must have same number "
        msg += "of rows. A has %i and B has %i rows currently." %(mat.Nr,rhs.Nr)
        raise Exception(msg)


    A = mat if destroy_mat else mat.copy()

    # For pivoting information.
    nr = nrc(A.Nr, A.Br, A.context.row, A.context.num_rows)
    ipiv = <int *>malloc(sizeof(int) * (nr+A.Br))

    pdgetrf_(&(A.Nr), &(A.Nc),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(),
             ipiv, &info)

    if info < 0:
        raise Exception("Something weird has happened in the LU decomposition.")
    elif info > 0:
        raise Exception("Solving with this decomp will cause division by 0.")

    # Factorization done. Now solve.

    B = rhs if destroy_rhs else rhs.copy()

    torno = "T" if trans else "N"

    pdgetrs_(torno, &(A.Nr), &(B.Nc),
             <double *>A._data(), &_ONE, &_ONE, A._getdesc(), ipiv,
             <double *>B._data(), &_ONE, &_ONE, B._getdesc(),
             &info)

    if info < 0:
        raise Exception("Something weird has happened in the LU solving.")
    elif info > 0:
        raise Exception("This should never occur.")

    return A,B
