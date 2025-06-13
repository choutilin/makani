'''
This module handles time as a float number,
as days since the _origin, 2000-Jan-01.

That is,
0.0 means 2000-Jan-01 00:00:00
1.5 means 2000-Jan-02 12:00:00

'''
from datetime import datetime, timedelta
from calendar import isleap as cisleap
from math import isnan, isinf, floor
from dateutil.parser import parse as parseDate


def example():
    # ---- #
    year0, month0, day0 = 2024, 3, 7 
    year1, month1, day1 = 2025, 7, 2

    date0 = ymd2float(year0, month0, day0)
    date1 = ymd2float(year1, month1, day1)

    deltaDate = date1 - date0

    strDate0 = float2format(date0, '%Y-%m-%d')
    strDate1 = float2format(date1, '%Y-%m-%d')

    print(f'{strDate1} is "{deltaDate}" days from {strDate0}')

    # ---- #
    date300DayLater = today() + 300
    day300DayLater = month(date300DayLater)

    date2MonthLater = addMonth(date0, 2)

    print(f'The month 300 days later is "{day300DayLater}".')
    print(f'The date 2 months later is "{float2format(date2MonthLater)}".')

    # ---- #
    f = ymd2float(2013, 4, 1)
    y, m, d = float2ymd(f)
    print((y, m, d))
    print(dayOfClim(f))



def _origin(): return datetime(2000, 1, 1)

def _float2datetime(f):
    if isinf(f) | isnan(f):
        return f
    return _origin() + timedelta(days=f)


def datetime2float(d): return (d - _origin()).total_seconds()/86400
def datetime2int(d): return int(datetime2float(d))
def ymd2float(*input): return datetime2float(datetime(*input))
def ymd2int(*input): return int(ymd2float(*input))
def float2ymd(f): return year(f), month(f), day(f)
def now(): return datetime2float(datetime.now())
def today(): return floor(datetime2float(datetime.today()))


def float2format(f, fmt='%Y%m%d'): return _float2datetime(float(f)).strftime(fmt)
def format2float(s, fmt): return datetime2float(format2datetime(s, fmt))
def format2datetime(s, fmt): return datetime.strptime(s, fmt)
def string2datetime(s): return parseDate(s)
def string2float(s): return datetime2float(parseDate(s))

def addMonth(f0, delta=1, warning=True):
    y, m, d = year(f0), month(f0)+delta, day(f0)
    remains = f0 % 1

    for i in range(9999):  # just to be safe
        if 1 <= m and m <= 12:
            break
        if 12 < m:
            y, m = y+1, m-12
        if m < 1:
            y, m = y-1, m+12

    dom = daysOfMonth(ymd2float(y, m, 1))
    if d > dom:
        if warning:
            print(f'Warning (addmonth): day is changed from {d} to {dom} for {y}-{m}-{d}')
        d = dom

    return ymd2float(y, m, d) + remains


def monthDelta(f0, f1):
    yr0, mm0, __ = float2ymd(f0)
    yr1, mm1, __ = float2ymd(f1)
    return (yr1-yr0)*12 + (mm1-mm0)


def daysOfMonth(f0):
    m = month(f0)
    if m in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif m in [4, 6, 9, 11]:
        return 30
    elif m == 2:
        if isleap(f0):
            return 29
        else:
            return 28
        

def dayOfYear(f): return _float2datetime(f).timetuple().tm_yday
def dayOfWeek(f): return _float2datetime(f).timetuple().tm_wday
def dayOfYear229(f):
    doy = dayOfYear(f)
    if doy <= 31 + 28: #Feb-28
        return doy
    if isleap(f):
        return doy
    return doy + 1 # skipped 229


def year(f): return _float2datetime(f).year
def month(f): return _float2datetime(f).month
def day(f): return _float2datetime(f).day
def hour(f): return _float2datetime(f).hour
def minute(f): return _float2datetime(f).minute
def second(f): return _float2datetime(f).second


def isleap(f): return cisleap(year(f))
def yearIsLeap(year): return cisleap(year)

def dayOfClim(f, keepDecimals=False):
    __, m, d, remains = *float2ymd(f), f % 1
    out = ymd2float(2000, m, d) - ymd2float(2000, 1, 1)
    if keepDecimals:
        out += remains
    return out
