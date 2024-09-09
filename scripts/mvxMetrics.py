# mvxMetrics.py - Compiles metrics of MVX performance: imaging & isolations (targeting, release, & collection/deposition).

from pathlib import Path
from scipy.io import loadmat
import numpy as np
from statistics import mean, stdev
import pandas as pd

# Total imaging scan time (s)
scan_time = None
scan_time_stdev = None
scan_times = []

# Targeting time per raft (s)
find_raft_time = None
find_raft_time_stdev = None
find_raft_times = []

# Needle release count per raft
needle_release_count = None
needle_release_count_stdev = None
needle_release_counts = []

# Needle release times per actuation (s)
needle_release_time = None
needle_release_time_stdev = None
needle_release_times = []

# Total time for collection/deposition per raft (s)
collection_time = None
collection_time_stdev = None
collection_times = []

# Total time from target ID to deposition per raft (s)
single_raft_time = None
single_raft_stdev = None
single_raft_times = []

## Set directories for raw data folders. ##
data_dir = [r'C:\Users\MVX\Documents\2 - Data\230504_H9C2N1_1.4&1.5e6_48hr_BMP4']

## Set total # arrays for each dir. ##
num_arrays = [2]

## Set # rafts isolated for each array. ##
num_rafts = [None]

# Compile data for all arrays in all raw data folders.
for i in range(len(data_dir)):
    dir_path = Path(data_dir[i])
    dir_subdata_path = dir_path / 'data'
    # dir_release_paths = [x for x in dir_path.iterdir() if (x.is_dir() and x[:11] == 'ReleaseData')]

    scan_time_mat = loadmat(dir_subdata_path / 'Scan_Time.mat')
    scan_time_mat = scan_time_mat['Scan_Time'][:, :, 1]
    scan_times += np.concatenate(scan_time_mat).ravel().tolist()

#     for dir_r in dir_release_paths:
#         for j in range(num_arrays[i]):
#             releasestatistics_mat = loadmat(dir_subdata_path / f'ReleaseStatistics at end for array {j}.mat')
#
#             findrafttimes_mat = releasestatistics_mat['FindRaftTimes']
#             count_mat = releasestatistics_mat['count']
#             needletimes_mat = releasestatistics_mat['needletimes']
#             collectiontimes_mat = releasestatistics_mat['CollectionTimes']
#             singlerafttimes_mat = releasestatistics_mat['SingleRaftTimes']
#
#             find_raft_times += np.concatenate(findrafttimes_mat).ravel().tolist()
#             needle_release_counts += np.concatenate(count_mat).ravel().tolist()
#             needle_release_times += np.concatenate(needletimes_mat).ravel().tolist()
#             collection_times += np.concatenate(collectiontimes_mat).ravel().tolist()
#             single_raft_times += np.concatenate(singlerafttimes_mat).ravel().tolist()
#
scan_time = mean(scan_times)
scan_time_stdev = stdev(scan_times)
print(scan_time)
print(scan_time_stdev)
#
# find_raft_time = mean(find_raft_times)
# find_raft_time_stdev = stdev(find_raft_times)
#
# needle_release_count = mean(needle_release_counts)
# needle_release_count_stdev = stdev(needle_release_counts)
#
# needle_release_time = mean(needle_release_times)
# needle_release_time_stdev = stdev(needle_release_times)
#
# collection_time = mean(collection_times)
# collection_time_stdev = stdev(collection_times)
#
# single_raft_time = mean(single_raft_times)
# single_raft_time_stdev = stdev(single_raft_times)
#
# metrix = ['Dir Paths',
#               'Num Arrays',
#               'Num Rafts',
#               'Scan Time per Array Mean (s)',
#               'Scan Time per Array Stdev (s)',
#               'Target Time per Raft Mean (s)',
#               'Target Time per Raft Stdev (s)',
#               'Needle Count per Raft Mean',
#               'Needle Count per Raft Stdev',
#               'Needle Time per Raft Mean (s)',
#               'Needle Time per Raft Stdev (s)',
#               'Collection Time per Raft Mean (s)'
#               'Collection Time per Raft Stdev (s)',
#               'Isolation Time per Raft Mean (s)',
#               'Isolation Time per Raft Stdev (s)']
# vals = [data_dir,
#         num_arrays,
#         num_rafts,
#         scan_time,
#         scan_time_stdev,
#         find_raft_time,
#         find_raft_time_stdev,
#         needle_release_count,
#         needle_release_count_stdev,
#         needle_release_time,
#         needle_release_time_stdev,
#         collection_time,
#         collection_time_stdev,
#         single_raft_time,
#         single_raft_time_stdev]
#
### Set directory for output csv ##
# outPath = Path(r'' / 'MVX Metrics.csv')
# df_metrix = pd.DataFrame({"Metrics": metrix,
#                         "Values": vals})
# df_metrix.to_csv(outPath, header=True, index=False)