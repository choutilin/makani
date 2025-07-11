#!/home/lkkbox/.conda/envs/rd/bin/python
from pytools import timetools as tt
from pytools import nctools as nct
from pytools import terminaltools as tmt
from pytools.fillNans2d import fillNans2d
from dataclasses import dataclass
import numpy as np
import os

@dataclass
class Variable:
    name: str
    standard_name: str
    level: int
    source: str
    ndim: int
    era5_varName: str = None


def main():
    for year in [2015]:
        run(year)

def run(year):
    """
    Create the yearly file that merges variables for training/validation.
    """
    # ---- settings
    overwrite = True
    partialWrites = ['u10', 'v10', 'mslp', 'sst']
    partialWrites = ['ssh']
    # debug_nt = -1
    numSmoothsNan = 20 * 4 # 20 deg * 4grid/deg
    variables = [
        Variable(
            name="u10",
            standard_name="u10",
            level=None,
            source="ERA5",
            ndim=3,
        ),
        Variable(
            name="v10",
            standard_name="v10",
            level=None,
            source="ERA5",
            ndim=3,
        ),
        Variable(
            name="mslp",
            standard_name="mslp",
            level=None,
            source="ERA5",
            ndim=3,
            era5_varName="msl",
        ),
        Variable(
            name="sst",
            standard_name="sst",
            level=None,
            source="ERA5",
            ndim=3,
        ),
        Variable(
            name="ssh",
            standard_name="ssh",
            level=None,
            source="HYCOM",
            ndim=3,
        ),
    ]

    # ---- auto settings
    outputPath = f'./data/{year}.nc'
    dtype = 'float32'
    TIME = [tt.ymd2float(year, 1, 1) + i * 0.25
            for i in range(1460 + 4*tt.isleap(tt.ymd2float(year, 1, 1)))]
    IVARIABLE = list(range(len(variables)))
    LAT = np.arange(-90, 90.25, 0.25)
    LON = np.arange(0, 360, 0.25)
    DIMNAMES = ['time', 'variable', 'latitude', 'longitude']
    nt, nv, ny, nx = len(TIME), len(variables), len(LAT), len(LON)
    shape = (nt, nv, ny, nx)
    fp = tmt.FlushPrinter()

    # ---- create the file
    if not overwrite and os.path.exists(outputPath):
        fp.print(f'File {outputPath} already exists. Use overwrite=True to overwrite.')
        return

    # create the file
    fp.print(f'Creating file {outputPath}, with {shape=}')
    nct.create(outputPath, 'fields', shape, DIMNAMES, dtype=dtype)
    for i, variable in enumerate(variables):
        nct.ncwriteatt(outputPath, 'fields', f'f{i:02d}', variable.standard_name)

    # write the dimension
    nct.write(outputPath, 'longitude', LON)
    nct.write(outputPath, 'latitude', LAT[::-1])
    nct.write(outputPath, 'variable', IVARIABLE)
    nct.write(outputPath, 'time', TIME)

    # ---- check the data source
    for iVar, variable in enumerate(variables):
        if partialWrites and not variable.name in partialWrites:
            fp.print(f'skipping {variable.name} for partial writes')
            continue

        path = getSrcPath(variable, TIME[0])
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        if variable.source == 'ERA5' and variable.era5_varName is not None:
            varName = variable.era5_varName
        else:
            varName = variable.standard_name

        srcShape = nct.getVarShape(path, varName)
        if [s for s in srcShape if s != 0] != [nt, ny, nx]:
            raise ValueError(f'{srcShape = }')

    # ---- loop over the variables
    for iVar, variable in enumerate(variables):
        if partialWrites and not variable.name in partialWrites:
            fp.print(f'skipping {variable.name} for partial writes')
            continue

        if variable.source == 'ERA5' and variable.era5_varName is not None:
            varName = variable.era5_varName
        else:
            varName = variable.standard_name

        fp.flushPrint(f'{variable.name} reading..')

        # read the data
        data, dims = nct.ncreadByDimRange(
            getSrcPath(variable, TIME[0]), varName,
            [[TIME[0], TIME[-1]], *[[None] * 2] * (variable.ndim - 1)],
        )
        data = np.squeeze(data)

        if not np.array_equal(dims[-1], LON):
            raise ValueError('Longitude dimension does not match')
        if not np.array_equal(dims[-2], LAT):
            raise ValueError('Latitude dimension does not match')

        # ssh's nan value
        if varName == 'ssh':
            data[(data > 15)] = np.nan

        # fill the nans
        if np.sum(np.isnan(data)) > 0:
            for it in range(nt):
                fp.flush(f'smoothing {it}/{nt}..')
                data[it, :] = fillNans2d(data[it, :], numSmoothsNan)

        # write the data
        slices = [
            slice(None),
            slice(iVar, iVar + 1),
            slice(None),
            slice(None),
        ]

        # flip lat to follow ecmwf convention
        data = np.reshape(data, (nt, 1, ny, nx))[:, :, ::-1, :]
        nct.write(outputPath, 'fields', data, slices)


def getSrcPath(variable, date):
    vn = variable.standard_name
    year = tt.year(date)
    if variable.source == 'HYCOM':
        return f'./source/HYCOM/{vn}/{vn}_{year}.nc'
    elif variable.source == 'ERA5':
        return f'./source/ERA5/{year}/ERA5_{vn}_{year}_6h.nc'
    else:
        raise ValueError(f'Unknown source {variable.source} for variable {variable.name}')


if __name__ == '__main__':
    main()