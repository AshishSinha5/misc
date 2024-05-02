from itertools import combinations
from multiprocessing import Manager, Pool, Process

s = "1 2 3 4\n5 1 1 8\n9 2 13 12\n13 14 15 16\n"

s = s.split("\n")
# s = [x.split(" ") for x in s]

def hash(pair):
    return int(pair[0]) + int(pair[1])

def process_line(pairs_hash_map_local,line):
    items = list(map(int, line.strip().split(' ')))  # Assuming items are integers
    
    # Use a set to store unique pairs
    unique_pairs = set(combinations(items, 2))
    
    # Update the pairs_hash_map_local
    for pair in unique_pairs:
        hashed_val = pair[0] + pair[1]
        pairs_hash_map_local[hashed_val] = pairs_hash_map_local.get(hashed_val, 0) + 1
    
    return pairs_hash_map_local

manager = Manager()
shared_pairs_hash_map = manager.dict()

p = Process(target=process_line, args=(shared_pairs_hash_map, s[0]))

p.start()

p.join()

print(shared_pairs_hash_map)


