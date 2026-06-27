# ratzarr
Rough mimic of GDAL's RasterAttributeTable in a Zarr file. Main aims are

  * Writing RAT columns directly to S3 bucket
  * Associating multiple RAT tables with a given raster, to allow grouping of related columns into individual RATs and avoid overly large single attribute tables.
  * Concurrent writing of separate columns into the same RAT, allowing computationally intensive columns to be computed in parallel

## Documentation
Module documentation is found [here](https://htmlpreview.github.io/?https://github.com/ubarsc/ratzarr/blob/main/docs/ratzarr.html).

Release notes for each version are [here](ReleaseNotes.md)

## Installation
Install from source

    pip install .

It requires the `zarr` package. Use with S3 also requires `s3fs` & `boto3`.
