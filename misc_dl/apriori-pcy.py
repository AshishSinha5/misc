import argparse
from itertools import combinations
from collections import defaultdict, Counter
import sys
# import time
from tqdm import tqdm
from multiprocessing import Pool

from functools import wraps
from time import time
import logging


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__,  te-ts))
        logging.info('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap


@timing
def preprocess_data(input_file):

    # data = defaultdict(list)
    prev_transaction_id = 1
    with open("processed_data.txt", 'w') as f2:
        with open(input_file, 'r') as f:
            for i, line in tqdm(enumerate(f)):
                if i < 3:
                    continue
                transaction_id = line.split()[0]
                item_id = line.split()[1]
                if transaction_id != prev_transaction_id:
                    f2.write("\n")
                    prev_transaction_id = transaction_id
                f2.write(item_id + " ")

def hash(x, y, bucket_field = 10000):
    # Hash function we were suggested to use
    
    hashed_val =  (int(x) ^ int(y)) % bucket_field
    # print(hashed_val)
    return hashed_val


def process_line_pass_1(line):
    items = line.split()
    unique_pairs = combinations(items, 2)
    pair_count = {}
    for pair in unique_pairs:
        hashed_val = hash(pair[0], pair[1])
        # print(hashed_val)
        if hashed_val not in pair_count:
            pair_count[hashed_val] = 1
        else:
            pair_count[hashed_val] += 1
    
    return pair_count




@timing
def pass1(min_sup:float):
    print("Pass 1")
    item_counts = defaultdict(int)
    pairs_hash_map = defaultdict(int)
    total_transactions = sum(1 for _ in open("processed_data.txt"))
    print(total_transactions)
    with open("processed_data.txt", 'r') as f:
        for i, line in enumerate(tqdm(f)):
            # skip first three lines of the file
            items = line.split()
            for item in items:
                item_counts[item] += 1

    print(f"{len(item_counts)} items")

    chunk_size = 2000
    with open("processed_data.txt", 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), chunk_size):
            f = lines[i:i+chunk_size]
            with Pool(4) as pool:
                results = pool.map(process_line_pass_1, iterable=f)
            print(len(results))
            for result in results:
                for hashed_val, count in result.items():
                    pairs_hash_map[hashed_val] += count

    # for result in results:
    #     for hashed_val, count in result.items():
    #         pairs_hash_map[hashed_val] += count
    print(len(pairs_hash_map))
    freq_items = {items:count for items, count in item_counts.items() if item_counts[items]/total_transactions >= min_sup}

    return freq_items, pairs_hash_map


@timing
def between_pass_1_2(freq_items, pairs_hash_map, min_sup:float):
    print("Between Pass 1 and 2")
    # maintain a bitmap of the pairs which are frequent or not
    bitmap = [0] * len(pairs_hash_map)
    for i, pair in enumerate(pairs_hash_map):
        if pairs_hash_map[pair] >= min_sup:
            bitmap[i] = 1

    freq_pair_count= {}
    for i in tqdm(range(0, len(freq_items))):
        for j in range(i+1, len(freq_items)):
            hashed_val = hash(freq_items[i], freq_items[j])
            if bitmap[hashed_val] == 1:
                # candidate pair requirement - 
                # both the items should be frequent
                # and the pair should hash to a frequent bucket
                freq_pair_count[(freq_items[i], freq_items[j])] = 0
    return freq_pair_count, bitmap

@timing
def pass2(freq_pair_count, bitmap, min_sup:float):
    print("Pass 2")
    with open("processed_data.txt", 'r') as f:
        for i, line in enumerate(tqdm(f)):
            items = line.split()
            for pair in combinations(items, 2):
                hashed_val = hash(pair[0], pair[1])
                if pair in freq_pair_count and bitmap[hashed_val] == 1:
                    freq_pair_count[pair] += 1

    num_transactions = i+1
    freq_pairs = {pair:count for pair, count in freq_pair_count.items() if freq_pair_count[pair]/num_transactions >= min_sup}
    sorter_freq_pairs = sorted(freq_pairs.items(), key=lambda x: x[1], reverse=True)
    # print top 5 pairs
    for i in range(5):
        print(sorter_freq_pairs[i])
        logging.info(sorter_freq_pairs[i])
    return freq_pairs


def main(args):
    min_sup = args.min_sup
    input_file = args.input_file
    preprocess_data(input_file)
    freq_items, pairs_hash_map = pass1(min_sup)
    freq_pair_count, bitmap = between_pass_1_2(freq_items, pairs_hash_map, min_sup)
    freq_pairs = pass2(freq_pair_count, bitmap, min_sup)

    







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apriori Algorithm')
    parser.add_argument('--input_file', type=str, help='Input file', default = "data/apriori/docword.enron.txt")
    parser.add_argument('--min_sup', type=float, help='Minimum support', default = 0.2)
    args = parser.parse_args()
    main(args)



