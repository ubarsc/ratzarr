"""
Routines for handling RAT-like columns stored in a Zarr array group.

The raster data model in GDAL includes a Raster Attribute Table (RAT).
Several formats support this, storing the data inside the same raster file.
However, sometimes it is not possible to write back into the file, such as
when the file is stored on AWS S3 storage.

This ratzarr module provides an alternative way to store RAT-like columns
alongside the raster file itself, in a way that allows extra columns to
be written incrementally (i.e. block-by-block), allowing for very large
tables to be built up, both in terms of large row count and large number of
columns. More columns can be added later, as required, and columns can be
deleted individually. Columns can also be resized to a new row count.

The RAT is stored as a set of 1-d arrays in a single Zarr array group
(https://zarr.dev/, https://github.com/zarr-developers/zarr-python). Currently,
ratzarr only makes use of the local disk and AWS S3 storage options. It does
not support zipfile storage, as this does not allow for writeable arrays on S3.

Simple usage:

    import ratzarr

    ratfile = 's3://mybucket/somepath/myrat.zarr'
    nRows = 1000000
    rz = ratzarr.RatZarr(ratfile)
    rz.setRowCount(nRows)

    # A very boring column
    col = numpy.arange(nRows) + 10

    rz.createColumn('BoringCol', col.dtype)
    rz.writeBlock('BoringCol', col, 0)

    # Read it all back
    col2 = rz.readBlock('BoringCol', 0, nRows)

For the RAT to be on local disk, the ``ratfile`` should just be an ordinary
path string.

For very large RATs, with large numbers of rows and possibly many columns,
it is often more memory-efficient to work with fixed-size blocks of data.
In this case, it is important to pay attention to the Zarr chunk size. The
default value is 500000, and it can be set before any columns are created
(see RatZarr.setChunkSize). Most importantly, the size of blocks being read
and/or written should be divisible by the chunk size. The simplest way is
often just to choose the block size as a multiple of the default Zarr chunk
size, but other considerations may require directly setting the Zarr chunk
size.

It is intended that all columns have the same length (i.e. number of rows), but
currently it is up to the user to enforce this.

"""
import sys
import os
import shutil
import unittest
from urllib.parse import urlparse

import numpy
import zarr
try:
    import boto3
except ImportError:
    boto3 = None


__version__ = "1.0.0"
CHUNKSIZE_ATTR = 'CHUNKSIZE'
DFLT_CHUNKSIZE = 500000


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
        serve no particular purpose in this context. All the usual numpy
        integer and float dtypes are supported, but Zarr only handles string
        arrays if they have dtype `numpy.dtypes.StringDType()`.

        This sort of RAT should not be used for storing things like the
        histogram or colour table. For this reason, we do not reproduce the
        concept of column usage, and all columns are effectively GFU_Generic.
        """
        self.usingS3 = False
        self.filename = filename
        components = urlparse(filename)
        self.usingS3 = (components.scheme == 's3')
        if self.usingS3:
            self.store = zarr.storage.FsspecStore.from_url(filename)
        else:
            self.store = filename
        self.grpName = "RAT"

        # First a sanity check if the store already exists.
        exists = RatZarr.exists(filename)
        isValid = RatZarr.isValidRatZarr(filename)

        if not exists and readOnly:
            msg = f"readOnly is True, but file '{filename}' does not exist"
            raise RatZarrError(msg)
        if not exists and not create:
            msg = f"File '{filename}' does not exist, but create is False"
            raise RatZarrError(msg)
        if exists and not isValid:
            msg = f"Zarr '{filename}' exists, but has no RAT group"
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

        self.chunksize = self.grp.attrs.get(CHUNKSIZE_ATTR, DFLT_CHUNKSIZE)
        if CHUNKSIZE_ATTR not in self.grp.attrs:
            self.grp.attrs[CHUNKSIZE_ATTR] = self.chunksize

        # Flags to prevent repetitious warning messages
        self.blockChunkMultipleWarningDone = False
        self.smallBlockWarningDone = False

    def setRowCount(self, rowCount):
        """
        Set the number of rows in the RAT.

        Usually do this before creating any columns. The row count persists
        between runs, so this only needs to be set when the RAT is first
        created, or if it needs to be changed.

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

    def getColumnNames(self):
        """
        Returns
        -------
          colNameList : list of str
            List of column names in RAT
        """
        return list(self.grp.keys())

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
        chunkshape = (self.chunksize,)
        a = self.grp.create_array(name=colName, dtype=dtype, shape=shape,
                                  chunks=chunkshape)
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

        For best efficiency, use a block length which is a multiple of the
        chunk size (see getRATChunkSize, getColumnChunkSize).

        If startRow+blockLen is larger than the column length, only
        the available rows are read, and the returned array has this smaller
        size.

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
        Write the given block of data into the named column, beginning at
        startRow. For best performance, use a block length which is a
        multiple of the chunk size (see setChunkSize, getRATChunkSize,
        getColumnChunkSize).

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
        # Check for chunk size warning conditions
        chunksize = self.columnCache[colName].chunks[0]
        if blockLen > chunksize and (blockLen % chunksize) != 0:
            msg = ("Warning: Block len {} not a multiple of " +
                   "chunk size {}").format(blockLen, chunksize)
            if not self.blockChunkMultipleWarningDone:
                print(msg, file=sys.stderr)
                self.blockChunkMultipleWarningDone = True
        if blockLen < chunksize and (startRow + blockLen) < self.rowCount:
            msg = ("Warning: Block len {} smaller than chunk " +
                   "size {}").format(blockLen, chunksize)
            if not self.smallBlockWarningDone:
                print(msg, file=sys.stderr)
                self.smallBlockWarningDone = True

        if (startRow + blockLen) > self.rowCount:
            msg = f"Current rowCount {self.rowCount} too small for block of "
            msg += f"length {blockLen} at startRow {startRow}"
            raise RatZarrError(msg)

        i1 = startRow
        i2 = startRow + blockLen
        self.columnCache[colName][i1:i2] = block

    def setChunkSize(self, chunksize):
        """
        Set the Zarr chunk size for all columns created after this call.
        This defaults to a sensible value, and should only be changed if
        you know what you are doing. Chunk sizes either too large or too
        small can have performance implications, so proceed with caution.

        The chunk size is preserved in the disk file, and will apply when
        next it is opened. Usually best to set this once, so that all
        columns have the same chunk size.

        When reading and/or writing block-by-block, it is strongly recommended
        that the block length be a multiple of the chunk size, to avoid
        significant performance degradation.

        Parameters
        ----------
          chunksize : int
            Number of rows per chunk
        """
        self.chunksize = chunksize
        if self.grp is not None:
            self.grp.attrs[CHUNKSIZE_ATTR] = chunksize

    def getRATChunkSize(self):
        """
        Return the chunk size for the RAT.
        """
        return self.grp.attrs.get(CHUNKSIZE_ATTR, DFLT_CHUNKSIZE)

    def getColumnChunkSize(self, colName):
        """
        Get the chunk size for the given column. Normally this is the same as
        the chunk size for the whole RAT, but if you suspect it is different,
        use this to check.

        Parameters
        ----------
          colName : str
            Name of column

        Returns
        -------
          chunkSize : int
            The chunk size for the given column (usually set when it was
            created)
        """
        self.openColumn(colName)
        chunks = self.columnCache[colName].chunks
        return chunks[0]

    @staticmethod
    def exists(zarrfile):
        """
        Check if the given filename exists. Does not confirm if it is a
        valid Zarr or RatZarr file.

        Parameters
        ----------
          zarrfile : str
            Full name of a possible zarr file (including 's3://' if required)

        Returns
        -------
          exists : bool
            True if the named file exists
        """
        components = urlparse(zarrfile)
        isS3 = (components.scheme == 's3')
        if isS3:
            if boto3 is None:
                raise RatZarrError('Using S3, but boto3 unavailable')

            s3client = boto3.client('s3')
            bucket = components.netloc
            key = components.path
            if key.startswith('/'):
                key = key[1:]
            response = s3client.list_objects(Bucket=bucket, Prefix=key)
            contents = response.get('Contents')
            fileExists = (contents is not None)
        elif components.scheme == '':
            path = components.path
            fileExists = os.path.exists(path)
        else:
            raise RatZarrError(f"Unknown zarrfile spec '{zarrfile}'")
        return fileExists

    @staticmethod
    def isValidRatZarr(zarrfile):
        """
        Check if the given filename is a valid RatZarr file

        Parameters
        ----------
          zarrfile : str
            Full name of a possible zarr file (including 's3://' if required)

        Returns
        -------
          isValid : bool
            True if the named file exists and is valid RatZarr
        """
        valid = RatZarr.exists(zarrfile)
        if valid:
            isZarr = RatZarr.exists(os.path.join(zarrfile, 'zarr.json'))
            isRatZarr = RatZarr.exists(os.path.join(zarrfile, 'RAT'))
            valid = (isZarr and isRatZarr)
        return valid

    @staticmethod
    def delete(zarrfile):
        """
        Delete the named RatZarr file.

        Silently returns if file does not exists. Raises exception if
        the file is not a valid RatZarr file.

        Parameters
        ----------
          zarrfile : str
            Full name of a possible zarr file (including 's3://' if required)
        """
        if not RatZarr.exists(zarrfile):
            return
        if not RatZarr.isValidRatZarr(zarrfile):
            raise RatZarrError(f"{zarrfile} is not valid RatZarr")

        components = urlparse(zarrfile)
        isS3 = (components.scheme == 's3')
        if isS3:
            if boto3 is None:
                raise RatZarrError('Using S3, but boto3 unavailable')

            s3client = boto3.client('s3')
            bucket = components.netloc
            key = components.path
            if key.startswith('/'):
                key = key[1:]

            response = s3client.list_objects(Bucket=bucket,
                                             Prefix=key)
            if 'Contents' in response:
                objectKeyList = [o['Key'] for o in response['Contents']]
                objSpec = {'Objects': [{'Key': k} for k in objectKeyList]}
                s3client.delete_objects(Bucket=bucket,
                                        Delete=objSpec)

            # Wait until it is actually gone
            while 'Contents' in response:
                response = s3client.list_objects(Bucket=bucket,
                                                 Prefix=key)
        elif components.scheme == '':
            shutil.rmtree(zarrfile)


class RatZarrError(Exception):
    """
    Generic exception for RatZarr
    """


# Code for unit tests
#
#


colNameByType = {
    numpy.int32: 'int32col',
    numpy.float32: 'float32col'
}
if hasattr(numpy.dtypes, "StringDType"):
    colNameByType[numpy.dtypes.StringDType()] = 'stringcol'


class AllTests(unittest.TestCase):
    """
    Run all tests
    """
    def test_simple(self):
        "Test reading/writing/creating a simple RAT"
        fn = 'test1.zarr'
        fullFilename = self.makeFilename(fn)
        self.deleteTestFile(fn)
        rz = RatZarr(fullFilename)
        n = 100
        rz.setRowCount(n)
        rz.setChunkSize(n // 2)
        for (dt, colName) in colNameByType.items():
            rz.createColumn(colName, dt)
            block = numpy.arange(n // 2).astype(dt)
            rz.writeBlock(colName, block, 0)
            rz.writeBlock(colName, block, n // 2)

            trueCol = numpy.concatenate([block, block])
            col = rz.readBlock(colName, 0, n)

            self.assertEqual(dt, col.dtype, 'dtype mis-match')
            numpy.testing.assert_array_equal(
                col, trueCol, f'Column data mis-match (dtype={block.dtype})')

        self.deleteTestFile(fn)

    def test_flags(self):
        "Test a bunch of exception conditions on constructor flags"
        fn = 'test1.zarr'
        fullFilename = self.makeFilename(fn)
        self.deleteTestFile(fn)

        # readOnly with non-existent file
        with self.assertRaises(RatZarrError):
            _ = RatZarr(fullFilename, readOnly=True)
        # create=False with non-existent file
        with self.assertRaises(RatZarrError):
            _ = RatZarr(fullFilename, create=False)

        self.deleteTestFile(fn)

    def test_resize(self):
        "Reset rowCount"
        fn = 'test1.zarr'
        fullFilename = self.makeFilename(fn)
        self.deleteTestFile(fn)

        rz = RatZarr(fullFilename)
        n = 100
        rz.setRowCount(n)
        c = numpy.arange(n).astype(numpy.int32)
        colName = 'col1'
        rz.createColumn(colName, c.dtype)
        rz.writeBlock(colName, c, 0)

        # Resize to smaller
        rz.setRowCount(n // 2)
        col = rz.readBlock(colName, 0, n)
        self.assertEqual(
            col.shape[0], (n // 2),
            'Truncated rowCount mis-match')

        # Resize bigger
        rz.setRowCount(n)
        col = rz.readBlock(colName, 0, n)
        self.assertEqual(col.shape[0], n, 'Increased rowCount mis-match')

        self.deleteTestFile(fn)

    def test_colnames(self):
        "Handling column names"
        fn = 'test1.zarr'
        fullFilename = self.makeFilename(fn)
        self.deleteTestFile(fn)

        rz = RatZarr(fullFilename)
        n = 100
        rz.setRowCount(n)
        c = numpy.arange(n).astype(numpy.int32)
        col1 = 'col1'
        col2 = 'col2'
        rz.createColumn(col1, c.dtype)
        rz.createColumn(col2, c.dtype)
        self.assertTrue(rz.colExists(col1), f"Column '{col1}' does not exist")
        colNameList = sorted(rz.getColumnNames())
        self.assertListEqual(
            colNameList, [col1, col2],
            'Incorrect list of columns')

        # Delete a column
        rz.deleteColumn(col1)
        self.assertFalse(rz.colExists(col1), f"Column '{col1}' not deleted")

        self.deleteTestFile(fn)

    def test_chunksize(self):
        "Chunk size manipulation"
        fn = 'test1.zarr'
        fullFilename = self.makeFilename(fn)
        self.deleteTestFile(fn)

        rz = RatZarr(fullFilename)
        rz.setRowCount(1000000)
        ratChunk = rz.getRATChunkSize()
        self.assertEqual(ratChunk, DFLT_CHUNKSIZE,
                         f'Unexpected chunk size {ratChunk}')
        newChunk = 100000
        rz.setChunkSize(newChunk)
        ratChunk = rz.getRATChunkSize()
        self.assertEqual(ratChunk, newChunk,
                         f'Chunk size not changed: {ratChunk}')

        colName = 'col1'
        rz.createColumn(colName, numpy.int32)
        colChunk = rz.getColumnChunkSize(colName)
        self.assertEqual(colChunk, newChunk,
                         f'Unexpected column chunk {colChunk}')

        rz2 = RatZarr(fullFilename)
        ratChunk = rz2.getRATChunkSize()
        self.assertEqual(ratChunk, newChunk,
                         'Chunk size not preserved on disk')

        self.deleteTestFile(fn)

    def makeFilename(self, filename):
        """
        Make full filename string
        """
        self.s3bucket = os.environ.get('S3BUCKET')
        if self.s3bucket is not None and boto3 is None:
            raise ValueError("S3BUCKET given, boto3 unavailable")
        self.usingS3 = (self.s3bucket is not None and boto3 is not None)
        if self.usingS3:
            fullFilename = f"s3://{self.s3bucket}/{filename}"
        else:
            fullFilename = filename
        return fullFilename

    def deleteTestFile(self, filename):
        """
        Delete the given test file, if it exists
        """
        RatZarr.delete(filename)


def mainCmd():
    unittest.main(module='ratzarr')


if __name__ == "__main__":
    mainCmd()
