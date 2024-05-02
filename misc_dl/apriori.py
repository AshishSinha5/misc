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


# add logging 
logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

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

@timing
def pass1(min_sup:float):
    print("Pass 1")
    item_counts = defaultdict(int)
    with open("processed_data.txt", 'r') as f:
        for line in tqdm(f):
            # skip first three lines of the file
            items = line.split()
            for item in items:
                item_counts[item] += 1
    num_transactions = len(item_counts)
    freq_items = {items:count for items, count in item_counts.items() if item_counts[items]/num_transactions >= min_sup}
    return freq_items

@timing
def between_pass_1_2(freq_items):
    print("Between Pass 1 and 2")
    freq_pair_count = {}
    for i in tqdm(range(0, len(freq_items))):
        for j in range(i+1, len(freq_items)):
            freq_pair_count[(freq_items[i], freq_items[j])] = 0
    return freq_pair_count

@timing 
def pass2(freq_pair_count, min_sup:float):
    print("Pass 2")
    with open("processed_data.txt", 'r') as f:
        for line in tqdm(f):
            items = line.split()
            for pair in combinations(items, 2):
                if pair in freq_pair_count:
                    freq_pair_count[pair] += 1
    num_transactions = len(freq_pair_count)
    freq_pairs = {pair:count for pair, count in freq_pair_count.items() if freq_pair_count[pair]/num_transactions >= min_sup}
    sorter_freq_pairs = sorted(freq_pairs.items(), key=lambda x: x[1], reverse=True)
    # print top 5 pairs
    for i in range(5):
        print(sorter_freq_pairs[i])
        logging.info(sorter_freq_pairs[i])
    return freq_pairs

@timing
def between_pass_2_3(freq_items, freq_pairs):
    # generating candidate triplets
    freq_triplet_count = {}
    for i in tqdm(range(0, len(freq_items))):
        for j in range(i+1, len(freq_items)):
            for k in range(j+1, len(freq_items)):
                if (freq_items[i], freq_items[j]) in freq_pairs and (freq_items[j], freq_items[k]) in freq_pairs and (freq_items[i], freq_items[k]) in freq_pairs:
                    freq_triplet_count[(freq_items[i], freq_items[j], freq_items[k])] = 0
    return freq_triplet_count

# @timing 
# def pass3(freq_triplet_count, min_sup:float):
#     with open("processed_data.txt", 'r') as f:
#         for line in tqdm(f):
#             items = line.split()
#             for triplet in combinations(items, 3):
#                 if triplet in freq_triplet_count:
#                     freq_triplet_count[triplet] += 1
#     num_transactions = len(freq_triplet_count)
#     freq_triplets = {triplet:count for triplet, count in freq_triplet_count.items() if freq_triplet_count[triplet]/num_transactions >= min_sup}
#     sorter_freq_triplets = sorted(freq_triplets.items(), key=lambda x: x[1], reverse=True)
#     # print top 5 triplets
#     for i in range(5):
#         print(sorter_freq_triplets[i])
#     return freq_triplets

def update_dict(line, freq_triplet_count):
        items = line.split()
        for triplet in combinations(items, 3):
            if triplet in freq_triplet_count:
                freq_triplet_count[triplet] += 1

def pass3(freq_triplet_count, min_sup:float):
    print("Pass 3")
    # add multiprocessing here
    with Pool() as pool:
        with open("processed_data.txt", 'r') as f:
            lines = []
            for line in f:
                lines.append(line)
            inputs =  zip(lines, [freq_triplet_count]*len(lines))
            pool.starmap(update_dict, tqdm(inputs, total = len(lines)))

    num_transactions = len(freq_triplet_count)
    freq_triplets = {triplet:count for triplet, count in freq_triplet_count.items() if freq_triplet_count[triplet]/num_transactions >= min_sup}
    sorter_freq_triplets = sorted(freq_triplets.items(), key=lambda x: x[1], reverse=True)
    # print top 5 triplets
    for i in range(5):
        print(sorter_freq_triplets[i])
        logging.info(sorter_freq_triplets[i])
    return freq_triplets



def main(input_file, min_sup):
    preprocess_data(input_file)
    freq_items = pass1(min_sup)
    freq_pair_count = between_pass_1_2(list(freq_items.keys()))
    freq_pairs = pass2(freq_pair_count, min_sup)
    freq_triplet_count = between_pass_2_3(list(freq_items.keys()), freq_pairs)
    freq_triplets = pass3(freq_triplet_count, min_sup)
    
    # find the association rules
    for triplet in freq_triplets:
        for pair in combinations(triplet, 2):
            if pair in freq_pairs:
                confidence = freq_triplets[triplet]/freq_pairs[pair]
                if confidence >= min_sup:
                    print(pair, "=>", triplet, confidence)
                    logging.info(pair, "=>", triplet, confidence)




if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Apriori algorithm for frequent itemset mining.")
    parser.add_argument("--input_file", type=str, help="Path to the input file.", default = "data/apriori/docword.enron.txt")
    # parser.add_argument("output_file", type=str, help="Path to the output file.")
    parser.add_argument("--min_sup", type=float, help="Minimum support threshold.", default = 0.2)
    args = parser.parse_args()

    main(args.input_file, args.min_sup)

    