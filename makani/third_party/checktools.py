from collections.abc import Iterable


def checkType(target, validTypes, codeName):
    # make it iterable for an easier life
    if not isinstance(validTypes, Iterable) or isinstance(validTypes, str):
        validTypes = [validTypes]

    # check None type
    if None in validTypes and target is None:
        return
    elif None in validTypes:
        validTypes.pop(validTypes.index(None))

    # check lambda type
    if 'lambda' in validTypes and isLambda(target):
        return
    elif 'lambda' in validTypes:
        validTypes.pop(validTypes.index('lambda'))

    # check general types
    if isinstance(target, tuple(validTypes)) and not (
        isinstance(target, bool) and bool not in validTypes
    ):
        return

    raise TypeError(  # failed
        f'{codeName} shout be type {validTypes}, (found={target}, {type(target)})'
    )

def isIterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def isLambda(target): return callable(target) and target.__name__ == '<lambda>'


def checkLambdaArgs(lambdaObj, validArgs, codeName=None, raiseError=True):
    args = lambdaObj.__code__.co_varnames
    if args == tuple(validArgs):
        return True  # pass
    elif not raiseError:
        return False
    else:
        raise ValueError(
            f'Lambda arguments for "{codeName}" shout be {validArgs}, (found={args})'
        )
