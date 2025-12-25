"""
Routines for handling RAT-like columns stored in a Zarr array group.

All rather experimental as yet. Intended to be used with rios.ratapplier,
and also with pyshepseg's per-segment stats calculation. Not sure
how widely it might be useful beyond that.

"""
import zarr


__version__ = "1.0.0"


class RatZarr:
    def __init__(self, filename):
        """
        This class is a rough equivalent of the GDAL RasterAttributeTable
        class, implementing a RAT-like structure with a Zarray group.
        It is not supposed to be a drop-in replacement, just something with
        somewhat similar functionality.

        The filename is a string. If it begins with 's3://', the Zarray will be
        on AWS S3, otherwise it represents a local directory. If it does not
        exist, it will be created.

        The type of each column is a numpy dtype. We do not reproduce GDAL's
        GFT_Integer/Float/String field types, mainly because they seemed to
        serve no particular purpose in this context.

        This sort of RAT should not be used for storing things like the
        histogram or colour table. For this reason, we do not reproduce the
        concept of column usage, and all columns are effectively GFU_Generic.

        QUESTIONS:
            - Should it create if not exist? Perhaps need a flag to over-ride?
            - Should I support zipped Zarrays? Should I support them
              on S3 (read-only)? (Is that even possible?)
        """
        self.filename = filename
        if filename.lower().startswith('s3://'):
            self.store = zarr.storage.FsspecStore.from_url(filename)
        else:
            self.store = filename
        self.grpName = "RAT"

        # First a sanity check if the store already exists
        existsWithoutRAT = False
        try:
            zarr.open(store=self.store, path=self.grpName, mode='r')
        except zarr.errors.GroupNotFoundError:
            existsWithoutRAT = True
        except FileNotFoundError:
            # Does not exist, so will be created
            pass

        if existsWithoutRAT:
            msg = f"Zarray '{filename}' exists, but has no RAT group"
            raise RatZarrError(msg)

        self.grp = zarr.open_group(store=self.store, path=self.grpName,
                                   mode="a")
        self.columnCache = {}

        # If there are already columns present, find the rowCount
        colNameList = list(self.grp.keys())
        if len(colNameList) > 0:
            colName = colNameList[0]
            self.openColumn(colName)
            self.rowCount = self.columnCache[colName].shape[0]
        else:
            self.rowCount = 0

    def setRowCount(self, rowCount):
        """
        Set the number of rows in the RAT.

        Usually do this before creating any columns.

        If there are existing columns, they will be resized to the new
        rowCount. If this less than current, existing columns will be
        truncated.
        """
        self.rowCount = rowCount

        # Force any existing columns to the new rowCount
        for colName in self.grp:
            self.openColumn(colName)
            self.columnCache[colName].resize((rowCount,))

    def getRowCount(self):
        """
        Return the current row count
        """
        return self.rowCount

    def colExists(self, colName):
        """
        Return True if the column already exists
        """
        return (colName in self.grp)

    def createColumn(self, colName, dtype):
        """
        Create a new column. Uses the currently active rowCount
        """
        if self.colExists(colName):
            raise RatZarrError(f"Column '{colName}' already exists")

        shape = (self.rowCount,)
        a = self.grp.create_array(name=colName, dtype=dtype, shape=shape)
        self.columnCache[colName] = a

    def deleteColumn(self, colName):
        """
        Delete the named column
        """
        del self.grp[colName]

    def openColumn(self, colName):
        """
        Open a cached connection to an existing array on disk.

        This will happen automatically when reading or writing, and users
        should not usually need to call this method.
        """
        if colName not in self.grp:
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

        To read the entire column, use startRow=0 and blockLen=self.rowCount.
        """
        self.openColumn(colName)
        i1 = startRow
        i2 = startRow + blockLen
        block = self.columnCache[colName][i1:i2]
        return block

    def writeBlock(self, colName, block, startRow):
        """
        Write the given block of data into the named column begining at
        startRow
        """
        self.openColumn(colName)

        blockLen = block.shape[0]

        if (startRow + blockLen) < self.rowCount:
            msg = f"Current rowCount {self.rowCount} too small for block of "
            msg += f"length {blockLen} at startRow {startRow}"
            raise RatZarrError(msg)

        i1 = startRow
        i2 = startRow + blockLen
        self.columnCache[colName][i1:i2] = block


class RatZarrError(Exception):
    """
    Generic exception for RatZarr
    """
