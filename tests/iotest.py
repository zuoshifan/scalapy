
import unittest
import glob
import os

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

# If not using the mpi bits yet.
#nproc = 1
#rank = 0
#class Comm(object):

#    def barrier(self):
#        pass
#comm = Comm()

import pyscalapack.core as pscore
from pyscalapack import npyutils, blockcyclic
#import npyutils as npyutils

# Number of processors per side on grid
# usually this should be square
npx = int(nproc**0.5)
npy = npx
pscore.initmpi(gridsize = [npx, npy], blocksize = [16, 16])

class TestIO(unittest.TestCase):

    def setUp(self):
        n = 200
        self.n = n
        # Make a matrix that to use as test data.
        # Non semetric and easy to check: A[i, j] = i + 6j + 5
        self.mat = (np.arange(n, dtype=np.float64)[:,None]
                    + 6 * np.arange(n, dtype=np.float64) + 5)
        self.mat.shape = (n, n)

    def test_read_header(self):
        if rank == 0:
            # Write a copy to disk using canned routines.
            np.save("tmp_test_origional.npy", self.mat)

            shape, fortran_order, dtype, offset = npyutils.read_header_data(
                    "tmp_test_origional.npy")
            self.assertEqual(shape, (self.n, self.n))
            self.assertFalse(fortran_order)
            self.assertEqual(dtype, '<f8')
            self.assertEqual(offset % 16, 0)
        comm.barrier()

    def test_write_header(self):
        if rank == 0:
            fname = "tmp_test_new_hdr.npy"
            
            # Find out how much space is needed for the header.
            header_data = npyutils.pack_header_data((self.n, self.n), False,
                                                    float)
            header_len =  npyutils.get_header_length(header_data)
            self.assertEqual(header_len % 4096, 0)
            # Make an empty file, big enough to hold the header only.
            fp = open(fname, 'w')
            fp.seek(40000 - 1)
            fp.write("\0")
            fp.close()
            
            # Write it and read it and make sure the data is right.
            npyutils.write_header_data(fname, header_data)
            shape, fortran_order, dtype, offset = npyutils.read_header_data(fname)
            self.assertEqual(shape, (self.n, self.n))
            self.assertFalse(fortran_order)
            self.assertEqual(dtype, '<f8')
            self.assertEqual(offset, header_len)

            # Make sure the file is the same size as before.
            fp = open(fname, 'r')
            fp.seek(0, 2)
            self.assertEqual(fp.tell(), 40000)
        comm.barrier()

    def test_read_fortran(self):
        fmat = np.asfortranarray(self.mat)
        if rank == 0:
            np.save("tmp_test_origional.npy", fmat)
        comm.barrier()

        Amat = pscore.DistributedMatrix.from_npy("tmp_test_origional.npy",
                                                 blocksize=(16, 16))
        Bmat = pscore.DistributedMatrix.fromarray(self.mat,
                                                  blocksize=(16, 16))
        self.assertTrue(pscore.matrix_equal(Amat, Bmat))

    def test_write_fortran(self):
        fmat = np.asfortranarray(self.mat)
        
        Dmat = pscore.DistributedMatrix.fromarray(fmat,
                                                  blocksize=(16, 16))
        Dmat.to_npy("tmp_test_origional.npy")
        if rank == 0:
            Bmat = np.load("tmp_test_origional.npy")
            self.assertTrue(np.isfortran(Bmat))
            self.assertTrue(np.allclose(Bmat, self.mat))

    def test_read_C(self):
        cmat = np.ascontiguousarray(self.mat)
        if rank == 0:
            np.save("tmp_test_origional.npy", cmat)
        comm.barrier()

        Amat = pscore.DistributedMatrix.from_npy("tmp_test_origional.npy",
                                                 blocksize=(16, 16))
        Bmat = pscore.DistributedMatrix.fromarray(self.mat,
                                                  blocksize=(16, 16))
        self.assertTrue(pscore.matrix_equal(Amat, Bmat))

    def test_write_C(self):
        cmat = np.ascontiguousarray(self.mat)
        
        Dmat = pscore.DistributedMatrix.fromarray(cmat,
                                                  blocksize=(16, 16))
        Dmat.to_npy("tmp_test_origional.npy", fortran_order=False)
        if rank == 0:
            Bmat = np.load("tmp_test_origional.npy")
            self.assertFalse(np.isfortran(Bmat))
            self.assertTrue(np.allclose(Bmat, self.mat))

    def tearDown(self):
        if rank == 0:
            files = glob.glob("tmp_test_*")
            for f in files:
                os.remove(f)
        comm.barrier()


class TestLargeFile(unittest.TestCase):
    """These tests make sure reading large files where each node uses more than
    2Gb of memory works."""
    
    def setUp(self):
        self.fname = '/dev/zero'
        # choose n such that each process has more than 2G of data.
        n = npx * 18000
        self.shape = (n, n)
        self.blocksize = [512, 512]
        self.m = pscore.DistributedMatrix(self.shape, blocksize=self.blocksize,
                                          dtype=np.dtype(float))
        self.m.local_array[...] = 5.

    def test_large_C(self):
        blockcyclic.mpi_readmatrix(self.fname, MPI.COMM_WORLD, self.shape, 
                np.dtype(float).type, self.blocksize, 
                (self.m.context.num_rows, self.m.context.num_cols),
                order='C', displacement=0, local_array=self.m.local_array)
        self.assertTrue(np.all(self.m.local_array == 0))

    def test_large_F(self):
        blockcyclic.mpi_readmatrix(self.fname, MPI.COMM_WORLD, self.shape, 
                np.dtype(float).type, self.blocksize, 
                (self.m.context.num_rows, self.m.context.num_cols),
                order='F', displacement=0, local_array=self.m.local_array)
        self.assertTrue(np.all(self.m.local_array == 0))


class TestChunkedRectangular(unittest.TestCase):

    def setUp(self):
        self.shape = (432, 653)
        self.mat = (np.arange(self.shape[0], dtype=np.float64)[:,None]
                    + 5 * np.arange(self.shape[1], dtype=np.float64)[None,:]
                    + 10)
        self.blocksize = [7, 5]
        self.m = pscore.DistributedMatrix(self.shape, blocksize=self.blocksize,
                                          dtype=np.dtype(np.float64))
        self.m.local_array[...] = 5.
    
    def test_C(self):
        cmat = np.ascontiguousarray(self.mat)
        if rank == 0:
            cmat.tofile("tmp_test_origional.npy")
        comm.barrier()

        ref_mat = pscore.DistributedMatrix.fromarray(self.mat,
                                                  blocksize=self.blocksize)
        blockcyclic.mpi_readmatrix("tmp_test_origional.npy", MPI.COMM_WORLD,
                self.shape, np.dtype(float).type, self.blocksize, 
                (self.m.context.num_rows, self.m.context.num_cols),
                order='C', displacement=0, local_array=self.m.local_array,
                max_single_read_size=600000)
        self.assertTrue(pscore.matrix_equal(self.m, ref_mat))
        self.assertTrue(np.allclose(self.m.local_array, ref_mat.local_array))
    
    def test_F(self):
        cmat = np.ascontiguousarray(self.mat)
        if rank == 0:
            cmat.T.tofile("tmp_test_origional.npy")
        comm.barrier()

        ref_mat = pscore.DistributedMatrix.fromarray(self.mat,
                                                  blocksize=self.blocksize)
        blockcyclic.mpi_readmatrix("tmp_test_origional.npy", MPI.COMM_WORLD,
                self.shape, np.dtype(float).type, self.blocksize, 
                (self.m.context.num_rows, self.m.context.num_cols),
                order='F', displacement=0, local_array=self.m.local_array,
                max_single_read_size=600000)
        self.assertTrue(pscore.matrix_equal(self.m, ref_mat))
        self.assertTrue(np.allclose(self.m.local_array, ref_mat.local_array))

    def tearDown(self):
        if rank == 0:
            files = glob.glob("tmp_test_*")
            for f in files:
                os.remove(f)
        comm.barrier()



if __name__ == '__main__' :
    unittest.main()

