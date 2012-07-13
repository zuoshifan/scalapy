import numpy as np

from mpi4py import MPI


# Map numpy type into MPI type
_typemap = { np.float32 : MPI.FLOAT,
            np.float64 : MPI.DOUBLE,
            np.complex128 : MPI.COMPLEX16 }


def ceildiv(x, y):
    """Round to ceiling division."""
    return ((int(x) - 1) / int(y) + 1)

def pid_remap(p, p0, P):
    return ((p + P - p0) % P)

def num_c_blocks(N, B):
    """Number of complete blocks globally."""
    return int(N / B)

def num_blocks(N, B):
    """Total number of blocks globally."""
    return ceildiv(N, B)

def num_c_lblocks(N, B, p, P):
    """Number of complete blocks locally."""
    nbc = num_c_blocks(N, B)
    return int(nbc / P) + int(1 if ((nbc % P) > p) else 0)


def num_lblocks(N, B, p, P):
    """Total number of local blocks."""
    nb = num_blocks(N, B)
    return int(nb / P) + int(1 if ((nb % P) > p) else 0)

def partial_last_block(N, B, p, P):
    """Is the last local block partial?"""
    return ((N % B > 0) and ((num_c_blocks(N, B) % P) == p))


def num_rstride(N, B, stride):
    """Length of block strided row."""
    return num_blocks(N, B) * stride


#size_t stride_page(size_t B, size_t itemsize) {
#
#  size_t pl;
#
#  pl = (size_t) sysconf (_SC_PAGESIZE) / itemsize;
#  return ceildiv(B, pl) * pl;
#}


#size_t num_rpage(size_t N, size_t B, size_t itemsize) {
#  return num_rstride(N, B, stride_page(B, itemsize));
#}


def numrc(N, B, p, P):
    """The number of rows/columns of the global array local to the process.
    """
    
    # Number of complete blocks owned by the process.
    nbp = num_c_lblocks(N, B, p, P)
    
    # Number of entries of complete blocks owned by process.
    n = nbp * B
    
    # If this process owns an incomplete block, then add the number of entries.
    if partial_last_block(N, B, p, P):
        n += N % B

    return n


def indices_rc(N, B, p, P):
    """The indices of the global array local to the process.
    """
    
    nt = numrc(N, B, p, P)
    nb = num_c_lblocks(N, B, p, P)

    ind = np.zeros(nt)

    ind[:(nb*B)] = ((np.arange(nb)[:, np.newaxis] * P + p)*B +
                    np.arange(B)[np.newaxis, :]).flatten()

    if (nb * B < nt):
        ind[(nb*B):] = (nb*P+p)*B + np.arange(nt - nb*P)

    return ind


def mpi_readmatrix(fname, comm, gshape, dtype, blocksize, process_grid,
                   order='F', displacement=0, local_array=None,
                   max_single_read_size=2**30):
    """Distribute a block cyclic matrix read from a file (using MPI-IO).

    The order flag specifies in which order (either C or Fortran) the array is
    on disk. Importantly the returned `local_array` is ordered the *same* way.
    
    Parameters
    ----------
    fname : string
        Name of file to read.
    comm : mpi4py.MPI.COMM
        MPI communicator to use. Must match with the one used by BLACS (if using
        Scalapack).
    gshape : (nrows, ncols)
        Shape of the global matrix.
    blocksize : (blockrows, blockcols)
        Blocking size for distribution.
    process_grid : (prows, pcols)
        The shape of the process grid. Must be the same total size as
        comm.Get_rank(), and match the BLACS grid (if using Scalapack).
    order : 'F' or 'C', optional
        Is the matrix on disk is 'F' (Fortran/column major), or 'C' (C/row
        major) order. Defaults to Fortran ordered.
    displacement : integer, optional
        Use a displacement from the start of the file. That is ignore the first
        `displacement` bytes.
    local_array : numpy array
        Array into which the read data will be copied.  Array must be the
        correct shape, but need not be the correct data type of ordering.  This
        argument is usefull for reformatting the read data and may also save
        memory for very large reads.
    max_single_read_size: integer
        Maximum size of a single read per process (in bytes).  Reads bigger
        than this will be split into multiple reads.  There is a known MPIIO
        bug that occurs if this is bigger than 2**31

    Returns
    -------
    local_array : np.ndarray
        The section of the array local to this process.  If the argument of the
        same name was supplied, this is just a reference to that array.
    """
    
    if dtype not in _typemap:
        raise Exception("Unsupported type.")

    # Get MPI type
    mpitype = _typemap[dtype]

    # Sort out F, C ordering
    if order not in ['F', 'C']:
        raise Exception("Order must be 'F' (Fortran) or 'C'")

    # Set file ordering
    mpiorder = MPI.ORDER_FORTRAN if order=='F' else MPI.ORDER_C 

    # Get MPI process info
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Check process grid shape
    if size != process_grid[0]*process_grid[1]:
        raise Exception("MPI size does not match process grid.")

    # Get shape of local segment
    process_position = [int(rank / process_grid[1]), int(rank % process_grid[1])]
    lshape = map(numrc, gshape, blocksize, process_position, process_grid)

    # Allowcate memory for the output.
    if local_array is None:
        local_array = np.empty(lshape, dtype=dtype, order=order)
    elif (local_array.shape[0] != lshape[0]
          or local_array.shape[1] != lshape[1]):
        msg = "Array supplied for output is the wrong shape."
        raise ValueError(msg)

    # We split the read into batches of rows (columns for fortran ordering) to
    # keep a single read from being to large. Reads bigger than 2GB crash due
    # to a bug in MPIIO.
    max_read_size = max_single_read_size // mpitype.Get_size()
    # Find out the global shape of the chunks matrix to read.
    if order is 'F':
        # In Fortran order we read a subset of the columns at a time.
        read_rows = gshape[1]
        read_cols = max_read_size // read_rows
        # Make read_cols divisable by (blocksize[1] * process_grid[1]).
        read_cols = read_cols - read_cols % (blocksize[1] * process_grid[1])
        read_gshape = (read_rows, read_cols)
    elif order is 'C':
        # In C order, we read a subset of the rows at a time.
        read_cols = gshape[1]
        read_rows = max_read_size // read_cols
        # Make read_rows divisable by (blocksize[0] * process_grid[0]).
        read_rows = read_rows - read_rows % (blocksize[0] * process_grid[0])
        read_gshape = (read_rows, read_cols)
    if read_gshape[0] <= 0 or read_gshape[1] <= 0:
        msg = "Blocksize too big, reads cannot be chuncked into small reads."
        raise RuntimeError(msg)
    
    # Loop over all the chunks of data to read (note that one of these loops is
    # only length 1 depending on Fortran or C ordering).
    f = MPI.File.Open(comm, fname, MPI.MODE_RDONLY)
    global_elements_read = 0
    l_rows_read = 0
    for ii in range(0, gshape[0], read_gshape[0]):
        l_cols_read = 0
        for jj in range(0, gshape[1], read_gshape[1]):
            read_rows = min(read_gshape[0], gshape[0] - ii)
            read_cols = min(read_gshape[1], gshape[1] - jj)
            this_read_gshape = (read_rows, read_cols)

            # Create distributed array view.
            darr = mpitype.Create_darray(size, rank, this_read_gshape,
                             [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                             blocksize, process_grid, mpiorder)
            darr.Commit()

            # Get shape of local segment
            this_lshape = map(numrc, this_read_gshape, blocksize, 
                              process_position, process_grid)
            
            # Check to see if darr agrees on the size of this read.
            if (this_lshape[0]*this_lshape[1]
                != darr.Get_size() / mpitype.Get_size()):
                raise RuntimeError("Strange mismatch is local shape size.")

            # Create buffer for data to be read into.
            read_array = np.empty(this_lshape, dtype=dtype, order=order)
            
            # Open the file, and read out the segments.
            # If there isn't at least one full block per process, this
            # segfaults for some reason.
            f.Set_view(displacement
                       + global_elements_read * mpitype.Get_size(),
                       mpitype, darr, "native")
            f.Read_all(read_array)

            local_array[l_rows_read:l_rows_read + read_array.shape[0],
                        l_cols_read:l_cols_read + read_array.shape[1]] = \
                    read_array

            global_elements_read += this_read_gshape[0] * this_read_gshape[1]
            l_cols_read += read_array.shape[1]
        if l_cols_read != local_array.shape[1]:
            msg = "Strange mismatch between local number of columns."
            raise RuntimeError(msg)
        l_rows_read += read_array.shape[0]
    if l_rows_read != local_array.shape[0]:
        msg = "Strange mismatch between local number of rows."
        raise RuntimeError(msg)
    f.Close()
    return local_array

    
def mpi_writematrix(fname, local_array, comm, gshape, dtype,
                    blocksize, process_grid, order='F', displacement=0):
    
    """Write a block cyclic distributed matrix to a file (using MPI-IO).

    The order flag specifies in which order (either C or Fortran) the array
    should be on on disk. Importantly the input `local_array` *must* be ordered
    in the same way.
    
    Parameters
    ----------
    fname : string
        Name of file to read.
    local_array : np.ndarray
        The array to write.
    comm : mpi4py.MPI.COMM
        MPI communicator to use. Must match with the one used by BLACS (if using
        Scalapack).
    gshape : (nrows, ncols)
        Shape of the global matrix.
    blocksize : (blockrows, blockcols)
        Blocking size for distribution.
    process_grid : (prows, pcols)
        The shape of the process grid. Must be the same total size as
        comm.Get_rank(), and match the BLACS grid (if using Scalapack).
    order : 'F' or 'C', optional
        Is the matrix on disk is 'F' (Fortran/column major), or 'C' (C/row
        major) order. Defaults to Fortran ordered.
    displacement : integer, optional
        Use a displacement from the start of the file. That is ignore the first
        `displacement` bytes.

    """
    
    if dtype not in _typemap:
        raise Exception("Unsupported type.")

    # Get MPI type
    mpitype = _typemap[dtype]


    # Sort out F, C ordering
    if order not in ['F', 'C']:
        raise Exception("Order must be 'F' (Fortran) or 'C'")

    mpiorder = MPI.ORDER_FORTRAN if order=='F' else MPI.ORDER_C 


    # Get MPI process info
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Check process grid shape
    if size != process_grid[0]*process_grid[1]:
        raise Exception("MPI size does not match process grid.")


    # Create distributed array view.
    darr = mpitype.Create_darray(size, rank, gshape,
                                 [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                                 blocksize, process_grid, mpiorder)
    darr.Commit()

    # Check to see if they type has the same shape.
    if local_array.size != darr.Get_size() / mpitype.Get_size():
        raise Exception("Local array size is not consistent with array description.")

    # Length of filename required for write (in bytes).
    filelength = displacement + gshape[0]*gshape[1]

    print filelength, darr.Get_size()

    # Open the file, and read out the segments
    f = MPI.File.Open(comm, fname, MPI.MODE_RDWR | MPI.MODE_CREATE)

    # Preallocate to ensure file is long enough for writing.
    f.Preallocate(filelength)

    # Set view and write out.
    f.Set_view(displacement, mpitype, darr, "native")
    f.Write_all(local_array)
    f.Close()


def bc_matrix_forward(src, blocksize, process, process_grid, order='F', dest=None):

    gshape = src.shape

    lshape = map(numrc, gshape, blocksize, process, process_grid)

    if dest is None:
        dest = np.empty(lshape, dtype=src.dtype, order='F')

    cblocks = map(num_c_blocks, gshape, blocksize)
    lcblocks = map(num_c_blocks, gshape, blocksize, process, process_grid)
    partial = map(partial_last_block, gshape, blocksize, process, process_grid)    

    clen = cblocks[0]*blocksize[0], cblocks[1]*blocksize[1]

    q1 = src[:clen[0], :clen[1]].reshape((blocksize[0], cblocks[0], blocksize[1], cblocks[1]))
    q2 = src[:clen[0], clen[1]:].reshape((blocksize[0], cblocks[0], -1))
    q3 = src[clen[0]:, :clen[1]].reshape((-1, blocksize[1], cblocks[1]))
    q4 = src[clen[0]:, clen[1]:].reshape((gshape[0] - clen[0], gshape[1] - clen[1]))
    
    #q1[:, process[0]::process_grid[0], :, process[1]::process_grid[1]].reshape()
        
