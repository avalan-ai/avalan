import sys
import os
import time

sys.path.append(os.getcwd())

from cluster.prepare_data import get_headers_pairs_list, write_dist_matrix
from cluster.token_edit_distance import get_distance_matrix

if len(sys.argv) < 3:
    print(
        "Too few arguments. You should provide: \n1. dataset_filename"
        + "\n2. output_data_filename"
    )
    sys.exit()

start = time.perf_counter()
dataset_filename_ = sys.argv[1]
output_data_filename_ = sys.argv[2]

headers_pairs = get_headers_pairs_list(dataset_filename_, verbose=True)

dist_matrix, max_dist = get_distance_matrix(
    list(map(lambda x: x[1], headers_pairs)), verbose=True
)

write_dist_matrix(dist_matrix, max_dist, output_data_filename_, verbose=True)

end = time.perf_counter()
