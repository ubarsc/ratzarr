"""
Routines for handling RAT-like columns stored in a Zarr array group.

All rather experimental as yet. Intended to be used with rios.ratapplier,
and also with pyshepseg's per-segment stats calculation. Not sure
how widely it might be useful beyond that.

"""
import os
import shutil
import unittest

import numpy
import zarr


__version__ = "1.0.0"


class RatZarr:
    def __init__(self, filename, readOnly=False, create=True):
        """
        This class is a rough equivalent of the GDAL RasterAttributeTable
        class, implementing a RAT-like structure with a Zarr group.
        It is not supposed to be a drop-in replacement, just something with
        somewhat similar functionality.

        Parameters
        ----------
          filename : str
            The Zarr file/store to use for the RAT. Can be a local
            path string, or a URL for S3, e.g. 's3://bucketname/path'
          readOnly : bool
            If True, the RAT cannot be modified
          create : bool
            If True (the default), the RAT will be created if it does not
            already exist

        The type of each column is a numpy dtype. We do not reproduce GDAL's
        GFT_Integer/Float/String field types, mainly because they seemed to
        serve no particular purpose in this context.

        This sort of RAT should not be used for storing things like the
        histogram or colour table. For this reason, we do not reproduce the
        concept of column usage, and all columns are effectively GFU_Generic.
        """
        self.filename = filename
        if filename.lower().startswith('s3://'):
            self.store = zarr.storage.FsspecStore.from_url(filename)
        else:
            self.store = filename
        self.grpName = "RAT"

        # First a sanity check if the store already exists
        existsWithoutRAT = False
        notExists = False
        try:
            zarr.open(store=self.store, path=self.grpName, mode='r')
        except zarr.errors.GroupNotFoundError:
            existsWithoutRAT = True
        except FileNotFoundError:
            notExists = True

        if existsWithoutRAT:
            msg = f"Zarr '{filename}' exists, but has no RAT group"
            raise RatZarrError(msg)
        if notExists and readOnly:
            msg = f"readOnly is True, but file '{filename}' does not exist"
            raise RatZarrError(msg)
        if notExists and not create:
            msg = f"File '{filename}' does not exist, but create is False"
            raise RatZarrError(msg)

        mode = "a"
        if readOnly:
            mode = "r"

        self.grp = zarr.open_group(store=self.store, path=self.grpName,
                                   mode=mode)
        self.columnCache = {}

        # If there are already columns present, find the rowCount
        colNameList = list(self.grp.keys())
        if len(colNameList) > 0:
            colName = colNameList[0]
            self.openColumn(colName)
            self.rowCount = self.columnCache[colName].shape[0]
        else:
            self.rowCount = 0

        self.chunksize = None
        self.shardfactor = None

    def setRowCount(self, rowCount):
        """
        Set the number of rows in the RAT.

        Usually do this before creating any columns.

        If there are existing columns, they will be resized to the new
        rowCount. If this less than current, existing columns will be
        truncated.

        Parameters
        ----------
          rowCount : int
            The desired number of rows in the RAT

        """
        self.rowCount = rowCount

        # Force any existing columns to the new rowCount
        for colName in self.grp:
            self.openColumn(colName)
            self.columnCache[colName].resize((rowCount,))

    def getRowCount(self):
        """
        Returns
        -------
          rowCount : int
            The current number of rows in the RAT

        """
        return self.rowCount

    def colExists(self, colName):
        """
        Return True if the column already exists

        Parameters
        ----------
          colName : str
            Name of column

        """
        return (colName in self.grp)

    def createColumn(self, colName, dtype):
        """
        Create a new column. Uses the currently active rowCount

        Parameters
        ----------
          colName : str
            Name of column
          dtype : Any numpy dtype
            The data type of the column to create

        """
        if self.colExists(colName):
            raise RatZarrError(f"Column '{colName}' already exists")

        shape = (self.rowCount,)
        if self.chunksize is not None:
            chunkshape = (self.chunksize,)
        else:
            chunkshape = "auto"
        if self.shardfactor is not None:
            if self.chunksize is None:
                msg = "Explicit shard factor requires explicit chunk size"
                raise RatZarrError(msg)
            shards = ((self.shardfactor * self.chunksize), )
        else:
            shards = None
        a = self.grp.create_array(name=colName, dtype=dtype, shape=shape,
                                  chunks=chunkshape, shards=shards)
        self.columnCache[colName] = a

    def deleteColumn(self, colName):
        """
        Delete the named column

        Parameters
        ----------
          colName : str
            Name of column

        """
        del self.grp[colName]

    def openColumn(self, colName):
        """
        Open a cached connection to an existing array on disk.

        This will happen automatically when reading or writing, and users
        should not usually need to call this method.

        Parameters
        ----------
          colName : str
            Name of column

        """
        if not self.colExists(colName):
            msg = f"Column {colName} not found in Zarr file {self.filename}"
            raise RatZarrError(msg)

        if colName not in self.columnCache:
            a = zarr.open_array(store=self.store,
                                path=f"{self.grpName}/{colName}")
            self.columnCache[colName] = a

    def readBlock(self, colName, startRow, blockLen):
        """
        Read a block from the given column. Return a numpy array of the
        block.

        To read the entire column, use startRow=0 and blockLen=rowCount.

        Parameters
        ----------
          colName : str
            Name of column
          startRow : int
            Row of first element to read (starts at 0)
          blockLen : int
            Number of rows to read

        Returns
        -------
          block : ndarray
            1-d numpy array of data for block

        """
        self.openColumn(colName)
        i1 = startRow
        i2 = startRow + blockLen
        block = self.columnCache[colName][i1:i2]
        return block

    def writeBlock(self, colName, block, startRow):
        """
        Write the given block of data into the named column, begining at
        startRow

        Parameters
        ----------
          colName : str
            Name of column
          block : ndarray
            1-d numpy array
          startRow : int
            Row of first element of block (starts at 0)

        """
        self.openColumn(colName)

        blockLen = block.shape[0]

        if (startRow + blockLen) > self.rowCount:
            msg = f"Current rowCount {self.rowCount} too small for block of "
            msg += f"length {blockLen} at startRow {startRow}"
            raise RatZarrError(msg)

        i1 = startRow
        i2 = startRow + blockLen
        self.columnCache[colName][i1:i2] = block

    def setChunkSize(self, chunksize):
        """
        Set the chunk size to use when creating columns. The default will
        allow the Zarr package to choose a chunk size.

        When using large RATs on S3, it is recommended that explicit chunksize
        and shardfactor be set.

        Parameters
        ----------
          chunksize : int
            Number of rows per chunk
        """
        self.chunksize = chunksize

    def setShardFactor(self, shardfactor):
        """
        Set the shardfactor. This is the number of chunks in each Zarr shard.
        The default behaviour is no sharding at all.

        Explicit sharding is particularly recommended with large RATs on S3.
        """
        self.shardfactor = shardfactor


class RatZarrError(Exception):
    """
    Generic exception for RatZarr
    """


# Code for unit tests
#
#


colNameByType = {
    numpy.int32: 'int32col',
    numpy.float32: 'float32col',
    numpy.dtypes.StringDType(): 'stringcol'
}


class AllTests(unittest.TestCase):
    """
    Run all tests
    """
    def test_simple(self):
        fn = 'test1.zarr'
        if os.path.exists(fn):
            shutil.rmtree(fn)
        rz = RatZarr(fn)
        n = 100
        rz.setRowCount(n)
        for (dt, colName) in colNameByType.items():
            rz.createColumn(colName, dt)
            block = numpy.arange(n // 2).astype(dt)
            rz.writeBlock(colName, block, 0)
            rz.writeBlock(colName, block, n // 2)

            trueCol = numpy.concatenate([block, block])
            col = rz.readBlock(colName, 0, n)

            self.assertEqual(dt, col.dtype, 'dtype mis-match')
            numpy.testing.assert_array_equal(col, trueCol,
                f'Column data mis-match (dtype={block.dtype})')

        shutil.rmtree(fn)


def mainCmd():
    unittest.main(module='ratzarr')


if __name__ == "__main__":
    mainCmd()
