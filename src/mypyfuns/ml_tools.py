from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np, pandas as pd
import fnmatch, os

def use_torch_cuda(use_cuda=True):
    if use_cuda is True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
         device = torch.device('cpu')

    print('Using device:', device)
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    return None

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def kmean_typing(df_in, nc=5, random_state=78787, print_result=False):
    """a selfmade kmean clutering function, with default nc=5, using minmaxscaler"""
    
    df_out, mm_scaler = scale_vars(df_in, MinMaxScaler(), fit=1)

    clustering = KMeans(n_clusters = nc, random_state = random_state).fit(df_out)

    inverse_var = inverse_scale_var(clustering.cluster_centers_, mm_scaler)

    if print_result is True:
        print(list(clustering.__dict__))
        print(inverse_var)
    
    return clustering

def scale_vars(var, scaler, fit=None): 
    """ variable need to be scaled, instantiated scaler, fit-true, then fit_transform, otherwise just transform
        if len(np.asarray(var).shape) == 1:
        var =np.expand_dims(var, axis=1)  """
    var = np.asarray(var)
    
    if fit is None:
        scaled_var = scaler.transform(var)
    else:
        scaled_var = scaler.fit_transform(var)
    
    return scaled_var, scaler

def inverse_scale_var(var, scaler):
    var = np.asarray(var)
    reversed_var = scaler.inverse_transform(var)
    return reversed_var

def remove_df_nans(df,cols): 
    """given a columns this function remove rows with nan values"""
    df = df.dropna(subset=cols)
    df = df.reset_index(drop = True)
    # sub_df = df.iloc[:,cols]
    # inds = pd.isnull(sub_df).any(1).nonzero()[0]
    # print('Error! The data has bad values at row(s)#{}'.format(inds))
    # print('should drop the rows with bad data using:')
    # print('df = df.drop(index = inds)')
    return df

def convert_str_to_num(dfarr):
    """ sometime the value in csv is in str format
        convert value from str to float """
    sz = len(dfarr)
    out_arr = np.zeros(sz)
    for i in range(0, sz):
        #print(i, dfarr[i])
        out_arr[i] = float(dfarr[i])
    return out_arr

def listdir(dirname, pattern='*'):
    return fnmatch.filter(os.listdir(dirname), pattern)


def get_cdf_pdf_stats(arr):
    s = pd.Series(arr, name = 'value')
    df = pd.DataFrame(s)
    # Frequency
    stats_df = df \
    .groupby('value') \
    ['value'] \
    .agg('count') \
    .pipe(pd.DataFrame) \
    .rename(columns = {'value': 'frequency'})

    # PDF
    stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

    # CDF
    stats_df['cdf'] = stats_df['pdf'].cumsum()
    stats_df = stats_df.reset_index()
    return stats_df


def calculate_cdf_series(arr):
    idx = np.arange(len(arr))
    st_idx = np.argsort(arr)
    dat = np.column_stack((idx,st_idx))
    dat = np.column_stack((dat,arr))
    # construct an original dataframe with original index and sort index
    df1 = pd.DataFrame(data=dat, columns=['ids','st_ids','values'])
    # get a new df by removing the rows with nans, and this will also remove both original and sort index of the set
    df2 = remove_df_nans(df1,['values'])
    # using the new df without nans, calculate the pdf and cdf, the result will be in a new dataframe
    df_stat = get_cdf_pdf_stats(arr)

    ##now we want to map the cdf back the original dataframe and taking care of the nans and ordering

    # first, we want to map the results from the df_stat back to df2 which has all nan values removed from df1
    sort_ids2 = np.argsort(df2['values'].values)
    nrow2 = len(df2['values'].values)
    argsort_ids2 = np.argsort(sort_ids2)

    # store the index that is not removed during nan removal, these indexes are the index in df1 which are not nans
    not_nan_ids = df2['ids'].astype(int)
    
    # store the df2 data and the stats in an array
    temp2 = np.zeros((nrow2,4))
    temp2[:,0] = df2['values'].values
    temp2[:,1] = df_stat['frequency'].values[argsort_ids2]
    temp2[:,2] = df_stat['pdf'].values[argsort_ids2]
    temp2[:,3] = df_stat['cdf'].values[argsort_ids2]

    # second, generate an nan matrix with the size of 4 columns and rows equals to df1 rows
    nrow1 = len(df1['values'].values)
    temp1 = np.zeros((nrow1,4)) * np.nan
    temp1[not_nan_ids,:] = temp2

    # # first genenerate a template array for frequency, pdf, and cdf, this array should be initialized with nans
    # temp = np.zeros((len(arr), 3)) * np.nan
    # # second we want to obtain the arg sorted index from the df2 which has all nan removed 
    # sort_idx = df2['st_ids'].astype(int)
    # # this way, the first entry in the df_stat = a indexed value in the original value, and that index is found in the first element in the sort_idx
    # for i in np.arange(len(sort_idx)):
    #     fv = df_stat['frequency'].values[i]
    #     pdfv = df_stat['pdf'].values[i]
    #     cdfv = df_stat['cdf'].values[i]
    #     i_temp = sort_idx[i]
    #     temp[i_temp,0] = fv; temp[i_temp,1] = pdfv; temp[i_temp,2] = cdfv
    
    # final_dat = np.zeros((len(arr),4))
    # final_dat[:,0] = arr
    # final_dat[:,1:] = temp
    col_names = ['values', 'frequency','pdf', 'cdf']
    f_df = pd.DataFrame(data=temp1, columns=col_names)

    return f_df

def describe_array(arr,col_name='input array'):
    arr = arr.flatten()
    dfar = pd.DataFrame(data=arr, columns=[col_name])
    return print(dfar.describe().T)