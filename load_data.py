
import pandas
import numpy as np
import time
import datetime

PK = 0
CITY = 1
DISTRICT = 2
CREATED_DATA = 3
TITLE = 4
DESC = 5
AREA = 6
BEDROOMS = 7
ADVERTISER = 8
SUBURBS = 9
LATITUDE = 10
LONGITUDE = 11
PRICE = 12

MILLION = 1000*1000

NORM_F_PRICE = 20
NORM_F_PK = 142320544
NORM_F_CITY = 710
NORM_F_DISTRICT = 673
NORM_F_DATE = 1700000000
NORM_F_AREA = 1450
NORM_F_BEDROOMS = 5
NORM_F_ADVERTISER = 2
NORM_F_SUBURBS = 2
NORM_F_LAT = 100
NORM_F_LONG = 420

city_centers = {}


def fill_city_centers(data):
    for ct in np.unique(data[:, CITY]):
        lat_ind = np.where(np.logical_and(data[:, CITY] == ct, (np.logical_not(np.isnan(data[:,  LATITUDE].astype(float))))))
        lng_ind = np.where(np.logical_and(data[:, CITY] == ct, (np.logical_not(np.isnan(data[:, LONGITUDE].astype(float))))))

        lt = data[lat_ind, LATITUDE]
        lg = data[lng_ind, LONGITUDE]

        if len(lt[0]) > 0:
            cc_lat = np.sum(lt) / len(lt[0])
        else:
            cc_lat = 0
        if len(lg[0]) > 0:
            cc_lng = np.sum(lg) / len(lg[0])
        else:
            cc_lng = 0

        city_centers[ct] = (cc_lat, cc_lng)


def fix_lat_long_data(data):
    print 'fix lat/long data'
    for ct in city_centers.keys():
        lat_ind = np.where(np.logical_and(data[:, CITY] == ct, np.isnan(data[:, LATITUDE].astype(float))))
        lng_ind = np.where(np.logical_and(data[:, CITY] == ct, np.isnan(data[:, LONGITUDE].astype(float))))

        data[lat_ind, LATITUDE] = city_centers[ct][0]
        data[lng_ind, LONGITUDE] = city_centers[ct][1]

    # if the city did not exist in 'city_centers':
    lat_ind = np.where(np.isnan(data[:, LATITUDE].astype(float)))
    lng_ind = np.where(np.isnan(data[:, LONGITUDE].astype(float)))
    data[lat_ind, LATITUDE] = 0
    data[lng_ind, LONGITUDE] = 0


def get_outliers_filter(data, output):
    to_remove = []
    for ct in np.unique(data[:, CITY]):
        inds = np.where(data[:, CITY] == ct)
        pcs = output[inds]
        pcs = np.sort(pcs)

        lb = len(pcs) * 0.05
        ub = len(pcs) * 0.95

        to_remove += np.where(np.logical_and(np.logical_or(output > ub, output < lb), data[:, CITY] == ct))


def load_all():
    filter = [TITLE, DESC, CREATED_DATA]
    [train_in, train_out] = load_data('train', True, filter)
    print '------------------------------------'
    fix_bedrooms(train_in)
    fix_nan_problem(train_in, DISTRICT, 0)
    fix_nan_problem(train_in, BEDROOMS, 2)
    fix_nan_problem(train_in, ADVERTISER, 1)
    fix_nan_problem(train_in, SUBURBS, 1)
    fix_price(train_out, train_in)
    normalize_data(train_in)
    fill_city_centers(train_in)
    fix_lat_long_data(train_in)

    prediction_in = load_data('test', False, filter)
    print '------------------------------------'
    fix_bedrooms(prediction_in)
    fix_nan_problem(prediction_in, DISTRICT, 0)
    fix_nan_problem(prediction_in, BEDROOMS, 2)
    fix_nan_problem(prediction_in, ADVERTISER, 1)
    fix_nan_problem(prediction_in, SUBURBS, 1)
    normalize_data(prediction_in)
    fix_lat_long_data(prediction_in)

    return [train_in, train_out, prediction_in]


def convert_time(t):
    return time.mktime(datetime.datetime.strptime(t, "%Y-%m-%d").timetuple())


def fix_bedrooms(data):
    print 'fix bedrooms'
    data[:, BEDROOMS] = np.log2(data[:, BEDROOMS].astype(float))


def fix_price(data, train_in):
    return
    print 'fix price'
    mw = (data[:] / MILLION) / train_in[:, AREA]
    data[:] = mw / NORM_F_PRICE


def unwrap_result(result, data):
    return result
    return result[:, 0] * NORM_F_PRICE * data[:, AREA] * MILLION


def normalize_data(data):
    return
    print 'normalizing data'
    data[:, PK]           = (data[:, PK]).astype(float)           / NORM_F_PK
    data[:, CITY]         = (data[:, CITY]).astype(float)         / NORM_F_CITY
    data[:, DISTRICT]     = (data[:, DISTRICT]).astype(float)     / NORM_F_DISTRICT
    data[:, AREA]         = (data[:, AREA]).astype(float)         / NORM_F_AREA
    data[:, BEDROOMS]     = (data[:, BEDROOMS]).astype(float)     / NORM_F_BEDROOMS
    data[:, ADVERTISER]   = (data[:, ADVERTISER]).astype(float)   / NORM_F_ADVERTISER
    data[:, SUBURBS]      = (data[:, SUBURBS]).astype(float)      / NORM_F_SUBURBS
    data[:, LATITUDE]     = (data[:, LATITUDE]).astype(float)     / NORM_F_LAT
    data[:, LONGITUDE]    = (data[:, LONGITUDE]).astype(float)    / NORM_F_LONG

    # for i in xrange(len(data)):
    #     data[i, CREATED_DATA] = convert_time(data[i, CREATED_DATA]) / NORM_F_DATE


def fix_nan_problem(data, dtype, substitute):
    inds = np.where(np.isnan(data[:, dtype].astype(float)))
    data[inds, dtype] = substitute


def load_data(dtype, split_output=False, filter_key=None):
    filename = 'data/' + dtype + '_data.csv'
    data_frame = pandas.read_csv(filename)

    print 'loading : \'%s\'' % filename
    print(data_frame.count())
    data_set = data_frame.values

    if filter_key is not None:
        data_set[:, filter_key] = 0

    valid_keys = range(data_frame.shape[1])
    if split_output:
        data_set_in = data_set[:, valid_keys[:-1]]
        data_set_out = data_set[:, valid_keys[-1]]
        return [data_set_in, data_set_out]
    else:
        return data_set[:, valid_keys]



