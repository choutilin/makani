'''
    varStruct = {
      'variableName': variable,
      'dim0Name': dim0,
      'dim1Name': dim1,
      ...
      'dimNName': dimN,
    }
'''
from . import checktools as chkt
import netCDF4 as nc
import numpy as np
import os


def getVarNames(fileName: str) -> list:
    try:
        with nc.Dataset(fileName, 'r') as h:
            varNames = list(h.variables.keys())
    except Exception as e:
        print(e)
        varNames = []
    return varNames


def getDimNames(fileName: str, varName: str) -> list:
    with nc.Dataset(fileName, 'r') as h:
        dimNames = list(h[varName].dimensions)
    return dimNames

def getVarUnits(fileName, varName):
    try:
        with nc.Dataset(fileName, 'r') as h:
            units = h[varName].units
    except Exception as e:
        print(e)
        units = []
    return units


def _errorIfFileExists(fileName):

    if not (os.path.isfile(fileName) | os.path.islink(fileName)):
        return

    yesNo = askYesNoRepeatedly(
        '[nctools]: file for writing already exists, do you want to continue?'
        + f'\n{fileName=}\n'
    )

    if yesNo != 'yes':
        raise FileExistsError(f'{fileName=}')


def _errorIfVariableExists(fileName, varName):

    if not (os.path.isfile(fileName) | os.path.islink(fileName)):
        return

    if not varName in getVarNames(fileName):
        return

    yesNo = askYesNoRepeatedly(
        '[nctools]: The variable already exists in file.'
        + 'Do you want to continue?'
        + f'\n{varName=}, {fileName=}\n'
    )

    if yesNo != 'yes':
        raise NameError(f'{varName=} in {fileName=}')


def _errorIfFileNotExists(fileName):
    if not os.path.exists(fileName):
        raise FileNotFoundError(f'{fileName=}')


def _errorIfVariableNotExists(fileName, varName):
    _errorIfFileNotExists(fileName)
    if varName not in getVarNames(fileName):
        raise ValueError(f'{varName=} not found in {fileName=}')


def askYesNoRepeatedly(message, numRepeats=3):
    for __ in range(numRepeats):
        yesNo = input(message + 'only accept "yes"/"no"\n')
        if yesNo in ['yes', 'no']:
            break
    return yesNo


def errorIfInconsistentExistingVariable(fileName, varStruct):

    if not (os.path.isfile(fileName) or os.path.islink(fileName)):
        return

    varName, *newDimNames = list(varStruct.keys())
    if varName not in getVarNames(fileName):
        return

    newShape = varStruct[varName].shape
    with nc.Dataset(fileName, 'r') as h:
        oldDimNames = list(h[varName].dimensions)
        if newDimNames != oldDimNames:
            raise ValueError(
                'The input dimension names are different than the existing file.\n'
                + f'{oldDimNames=}\n{newDimNames=}'
            )
        if newShape != h[varName].shape:
            raise ValueError(
                'The input variable has a different shape than the existing file.\n'
                + f'oldShape = {h[varName].shape}\n{newShape=}'
            )
    return


def errorIfNotASubsetOfTheExistingVariable(fileName, varStruct):

    if not (os.path.isfile(fileName) or os.path.islink(fileName)):
        return

    varName, *newDimNames = list(varStruct.keys())
    if varName not in getVarNames(fileName):
        return

    newShape = varStruct[varName].shape
    with nc.Dataset(fileName, 'r') as h:
        oldDimNames = list(h[varName].dimensions)
        if newDimNames != oldDimNames:
            raise ValueError(
                'The input dimension names are different than the existing file.\n'
                + f'{oldDimNames=}\n{newDimNames=}'
            )
        if newShape != h[varName].shape:
            raise ValueError(
                'The input variable has a different shape than the existing file.\n'
                + f'oldShape = {h[varName].shape}\n{newShape=}'
            )
    return

def create(fileName, varName, shape, dimNames, use_my_attrs=True,
           significant_digits=None, dtype=np.float32):
    # ---- validate inputs
    chkt.checkType(fileName, str, 'fileName')
    chkt.checkType(varName, str, 'varName')
    chkt.checkType(shape, [tuple, list], 'shape')
    chkt.checkType(dimNames, [tuple, list], 'dimNames')
    chkt.checkType(use_my_attrs, bool, 'use_my_attrs')
    chkt.checkType(significant_digits, [None, int], 'significant_digits')
    [chkt.checkType(d, int, 'elements in shape') for d in shape]
    [chkt.checkType(d, str, 'elements in dimNames') for d in dimNames]
    if len(shape) != len(dimNames):
        raise ValueError(
            'The number of dimensions in shape and dimNames are different.'
            + f'{len(shape)=}, {len(dimNames)=}'
        )

    # ---- casting shapes to tuple and dimNames to list
    shape = tuple(shape)
    dimNames = list(dimNames)

    # ---- check the consistency of shape and dimnames if variable already exists
    if os.path.exists(fileName):
        if varName in getVarNames(fileName):
            existingShape = getVarShape(fileName, varName)
            if shape != existingShape:
                raise ValueError(
                    'The shape of the variable is different from the existing file.\n'
                    + f'{existingShape=}, {shape=}'
                )
            existingDimNames = getDimNames(fileName, varName)
            if dimNames != existingDimNames:
                raise ValueError(
                    'The dimension names are different from the existing file.\n'
                    + f'{existingDimNames=}, {dimNames=}'
                )
            return # variable already exists and passed the check

    # ---- create the dimensions
    doDim = False
    if len(dimNames) > 1:
        doDim = True
    elif dimNames[0] != varName:
        doDim = True

    if doDim:
        for dimName, dimLength in zip(dimNames, shape):
            create(fileName, dimName, [dimLength], [dimName], use_my_attrs,
                significant_digits=None, dtype=np.float32)

    # ---- create the variable
    with nc.Dataset(fileName, 'a') as h_file:
        if len(shape) == 1 and varName == dimNames[0]:
            h_file.createDimension(dimNames[0], shape[0])
        h_file.createVariable(
            varName, dtype, dimNames, significant_digits=significant_digits, compression='zlib', complevel=9, shuffle=True
        )
        if use_my_attrs:
            set_my_attrs(h_file, dimNames)

    return


def save(fileName, varStruct, overwrite=False, use_my_attrs=True, significant_digits=None):

    if not overwrite:
        if os.path.isfile(fileName):
            n = 0
            while n < 10:
                n += 1
                yn = input(
                    f'nctools: file for saving already exists, do you want to overwrite it?\n{fileName}\n')
                if yn == 'yes':
                    break
                elif yn == 'no':
                    raise Exception('stop')
                else:
                    print('only accept yes/no', flush=True)
            if yn != 'yes':
                raise Exception('stop')

    errorIfInvalidVarStruct(varStruct)

    names = list(varStruct.keys())
    values = list(varStruct.values())
    values = [np.array(v) for v in values]

    varName = names[0]
    varvalue = values[0]

    if len(names) == 1:
        dimnames = [names[0]]
        dimvalues = [values[0]]
    else:
        dimnames = names[1:]
        dimvalues = values[1:]

    with nc.Dataset(fileName, 'a') as h_file:
        # check name exists
        existed_names = list(h_file.variables.keys())

        # check dimension exists
        for dimname, dimvalue in zip(dimnames, dimvalues):
            if dimname in existed_names:
                if dimvalue.shape != h_file[dimname].shape:
                    raise Exception(f'the dimension {dimname} has different shape than in the existing file.\n{fileName}')
        # check varaible exists
        if varName in existed_names:
            if varvalue.shape != h_file[varName].shape:
                raise Exception(
                    f'the variable {varName} has different shape than in the existing file.\n{fileName}')
            if dimnames != list(h_file[varName].dimensions):
                raise Exception(f'the variable {varName} has different dimension names than in the existing file.\n{fileName}')

        # create dimensions
        if len(dimnames) == 1 and dimnames[0] == varName:
            dimVarPairs = zip([varName], [varvalue])
        else:
            dimVarPairs = zip([*dimnames, varName], [*dimvalues, varvalue])
        for name, value in dimVarPairs:
            if name in existed_names:
                continue
            if name in dimnames:
                h_file.createDimension(name, len(value))
                h_file.createVariable(
                    name, np.float32, (name,), compression='zlib', complevel=9, shuffle=True)
                continue
            if name == varName:
                h_file.createVariable(
                    name, np.float32, dimnames, significant_digits=significant_digits, compression='zlib', complevel=9, shuffle=True)
                continue

        # write values
        for n, v in zip(dimnames, dimvalues):
            h_file[n][:] = v
        h_file[varName][:] = varvalue

        # set attributes
        if use_my_attrs:
            set_my_attrs(h_file, dimnames)
        return


def errorIfInvalidVarStruct(varStruct):
    '''
    varStruct = {
      'variableName': variable,
      'dim0Name': dim0,
      'dim1Name': dim1,
      ...
      'dimNName': dimN,
    }
    '''

    if not isinstance(varStruct, dict):
        raise TypeError('varStruct must be of a dictionary type')

    values = list(varStruct.values())
    for pos, value in enumerate(values):
        if not chkt.isIterable(value):
            raise TypeError(f'value at position {pos} is not iterable')

    values = [np.array(v) for v in values]

    var_shape = list(values[0].shape)
    if len(var_shape) == 1:
        dim_shape = [len(values[0])]
    else:
        dim_shape = [len(v) for v in values[1:]]

    # check consistency
    if var_shape != dim_shape:
        raise ValueError(f'shape of (var) & (dim) are inconsistent: {var_shape} & {dim_shape}')

    return True


def set_my_attrs(h_file, dimnames):
    for name in dimnames:
        h = h_file[name]
        if name.lower() in ['lon', 'longitude']:
            h.axis = 'X'
            h.units = 'degrees_east'
            h.long_name = 'longitude'
            h.standard_name = 'longitude'
        if name.lower() in ['lat', 'latitude']:
            h.axis = 'Y'
            h.units = 'degrees_north'
            h.long_name = 'latitude'
            h.standard_name = 'latitude'
        if name.lower() in ['lev', 'level', 'plev']:
            h.axis = 'Z'
        if name.lower() in [
            'time', 'valid_time', 'date', 'valid_date',
            *[f'time{i}' for i in range(10)],
            *[f'time_{i}' for i in range(10)],
        ]:
            h.axis = 'T'
            h.units = 'days since 2000-01-01 00:00:00'
            h.long_name = 'time'
            h.standard_name = 'time'



# def write(path, varName, value):
#     with nc.Dataset(path, 'a') as h_file:
#         h_file[varName][:] = value


def ncwriteatt(path, varName, attName, attValue):
    with nc.Dataset(path, 'a') as h_file:
        if varName == '/':
            h_file.setncattr(attName, attValue)
        else:
            setattr(h_file[varName], attName, attValue)

def write(fileName: str, varName: str, data, slices=None) -> str:
    with nc.Dataset(fileName, 'a') as hFile:
        if slices is None:
            hFile[varName][:] = data
        else:
            hFile[varName][slices] = data


def read(fileName, varName):
    _errorIfFileNotExists(fileName)
    _errorIfVariableNotExists(fileName, varName)
    try:
        with nc.Dataset(fileName, 'r') as h:
            data = h[varName][:]
            data = np.array(data)
    except RuntimeError as e:
        print(e)
        raise RuntimeError(f'{fileName = }, {varName = }')
    except Exception as e:
        print(e)
        raise RuntimeError('')
    return data


def getVarShape(fileName, varName):
    if not (os.path.isfile(fileName) or os.path.islink(fileName)):
        return None
    if varName not in getVarNames(fileName):
        return None
    with nc.Dataset(fileName, 'r') as h:
        shape = h[varName].shape
    return shape


def getVarDimLength(fileName, varName, iDim):
    shape = getVarShape(fileName, varName)
    if shape is None:
        return 0
    if len(shape)-1 < iDim:
        return 0
    return shape[iDim]


def ncreadatt(fileName: str, varName: str, attName: str) -> str:
    with nc.Dataset(fileName, 'r') as hFile:
        if varName == '/':
            hVar = hFile
        else:
            hVar = hFile[varName]
        try:
            attValue = hVar.getncattr(attName)
        except Exception as e:
            attValue = None
            print(e)
    return attValue


def ncreadtime(
    fileName: str, varName: str = 'time', attName: str = 'units'
) -> np.array:

    from . import timetools as tt

    timeValue = read(fileName, varName)
    timeUnits = ncreadattt(fileName, varName, attName).lower()

    # timeUnits = "{timeDelta}{delimitter}since{delimitter}{timeOrigin}"
    # parse time units -> timeDelta & timeOrigin
    if 'since' not in timeUnits:
        raise ValueError(f'cannot find "since" in {timeUnits=} to parse.')

    found = False
    validDelimitters = [' ', '_']
    for delimitter in validDelimitters:
        if timeUnits.split(delimitter)[1] == 'since':
            timeUnits = timeUnits.split(delimitter)
            found = True
            break
    if not found:
        raise ValueError(f'unable to parse {timeUnits=}')

    strTimeDelta = timeUnits[0]
    strTimeOrigin = delimitter.join(timeUnits[2:])

    timeOrigin = tt.string2float(strTimeOrigin)
    TIMEDELTA = {
        'second': 1/86400,
        'seconds': 1/86400,
        'minute': 1/1440,
        'minutes': 1/1440,
        'hour': 1/24,
        'hours': 1/24,
        'day': 1,
        'days': 1,
    }

    STRMONTH = ['month', 'months']
    STRYEAR = ['year', 'years']

    if strTimeDelta not in [*TIMEDELTA.keys(), *STRMONTH, *STRYEAR]:
        raise ValueError(f'unalbe to recognize {strTimeDelta=}')

    if strTimeDelta in STRMONTH:
        time = [tt.addMonth(timeOrigin, v) for v in timeValue]
    elif strTimeDelta in STRYEAR:
        time = [tt.addMonth(timeOrigin, v*12) for v in timeValue]
    else:
        time = [timeOrigin + v*TIMEDELTA[strTimeDelta] for v in timeValue]

    return np.array(time)


def ncread(fileName: str, varName: str, slices: list[slice] = None) -> np.array:

    _errorIfVariableNotExists(fileName, varName)

    # ---- checking slices ---- #
    if (slices is not None) and (not isinstance(slices, list)):
        raise TypeError('"slices" must be the list type')

    if slices is not None:

        for s in slices:
            if not isinstance(s, slice):
                raise TypeError('elements in "slices" must be the slice type.')

        with nc.Dataset(fileName, 'r') as h:
            ndim = h[varName].ndim
        if len(slices) != ndim:
            raise ValueError(
                f'The number of inquired dimensions (n={len(slices)}) '
                f'are different from the file (n={ndim}).'
            )

    # ---- read file ---- #
    with nc.Dataset(fileName, 'r') as h:
        if slices is None:
            data = h[varName][:]
        else:
            data = h[varName][slices]

    return np.array(data)


def ncreadByDimRange(
    fileName: str, varName: str, minMaxs: list[list],
    iDimT: int = None, decodeTime=True
):
    from .caltools import value2Slice

    #
    # ---- check file can be opened
    _errorIfVariableNotExists(fileName, varName)
    with nc.Dataset(fileName, 'r') as h:
        NDIM = h[varName].ndim
    dimNames = getDimNames(fileName, varName)

    #
    # ---- checking input types 
    if not isinstance(minMaxs, list):
        raise TypeError('"minMaxs" must be a list.')

    for minMax in minMaxs:
        if not isinstance(minMax, list):
            raise TypeError('The elements in "minMaxs" must be a list."')
        if len(minMax) != 2:
            raise ValueError('The minMax in "minMaxs" must have 2 elements')
        for minOrMax in minMax:
            if minOrMax is None:
                continue
            if not isinstance(minOrMax, (int, float)):
                raise ValueError(
                    'The min or Max in "minMaxs" must be a number')

    if not decodeTime and iDimT is not None:
        raise ValueError('"decodeTime" is false but "iDimT" is assigned.')

    if iDimT is not None:
        if not isinstance(iDimT, int):
            raise ValueError('"iDimT" must be None or an integer.')
        elif iDimT < 0 or iDimT >= NDIM:
            raise ValueError(f'"iDimT" must be >=0 and <= "NDIM" ({NDIM})')

    if decodeTime not in [True, False]:
        raise ValueError('"decodeTime" only accepts "True" or "False".')
    
    #
    # ---- check numDims
    numDims = len(dimNames)
    if len(minMaxs) != numDims:
        raise ValueError(f'incorrect number ({len(minMaxs)}) of minMaxs, {dimNames=}')

    #
    # ---- begins ---- #
    #
    #
    # ---- assign iDimT ---- #
    if iDimT is None and decodeTime:
        # guessing time's dimension name
        found = False
        dimNamesLower = [vn.lower() for vn in dimNames]
        for timeName in [
            'time', 'valid_time',
            *[f'time{i}' for i in range(10)],
            *[f'time_{i}' for i in range(10)],
        ]:
            if timeName in dimNamesLower:
                found, iDimT = True, dimNamesLower.index(timeName)
                break

        if not found:
            raise ValueError(
                f'unable to determine which one is time dimensions from {dimNames=}. '
                f'Assign "iDimT" manually or set "decodeTime" to false.'
            )
        
    #
    # ---- get slices for dimensions ---- #
    dimensions = [  # read dimensions
        read(fileName, dimName) # general dimensions
        if (iDim != iDimT) or (not decodeTime)
        else ncreadtime(fileName, dimName) # time dimension
        for iDim, dimName in enumerate(dimNames)
    ]

    dimensionsFlipped, dimsAreReversed = zip(*[ 
        (dimension[::-1], True)  # reverse the dimension if decreasing
        if dimension[0] > dimension[-1]
        else (dimension, False)
        for dimension in dimensions
    ])

    slicesFlipped = [ # determine the slice by minMaxs
        value2Slice(dimension, *minMax)
        for dimension, minMax in zip(dimensionsFlipped, minMaxs)
    ]

    dimensionsFlipped = [
        np.array(dim[sli]) 
        for dim, sli in zip(dimensionsFlipped, slicesFlipped)
    ]

    slices = [  # reverse the slice if needed
        slice(len(dim)-sli.stop, len(dim)-sli.start) 
        if rev else sli
        for sli, rev, dim in zip(slicesFlipped, dimsAreReversed, dimensions)
    ]

    # extract the dimension for output
    dimensions = [np.array(dim[sli]) for sli, dim in zip(slices, dimensions)]

    # read variable
    with nc.Dataset(fileName, 'r') as h:
        data = h[varName][slices]
    
    data = np.array(data)
    data = np.flip(data, axis=[iax for iax, rev in enumerate(dimsAreReversed) if rev])

    return data, dimensionsFlipped
