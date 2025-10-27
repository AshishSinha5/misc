import mmh3

class HashedFeature:
    """
    A class to represent a hashed feature using MurmurHash3.

    Attributes:
        feature_name (str): The name of the feature.
        num_buckets (int): The number of buckets to hash into.
    """

    def __init__(self, feature_name: str, num_buckets: int):
        self.feature_name = feature_name
        self.num_buckets = num_buckets

    def hash(self, value: any) -> int:
        """
        Hashes the input value into one of the buckets.

        Args:
            value (any): The input value to be hashed.

        Returns:
            int: The bucket index for the hashed value.
        """
        # convert the value to string for hashing
        value = str(value)
        # take the murmurhash3 of the value
        hash_value = mmh3.hash(value)
        # map the hash value to one of the buckets
        bucket_index = hash_value % self.num_buckets
        # take the absolute value to avoid negative indices
        bucket_index = abs(bucket_index)
        return bucket_index
    
# Example usage
if __name__ == "__main__":
    sample_values = ["JFK", "LAX", "ORD", "DFW", "DEN", "SFO", "ATL", "SEA", "MIA", "BOS"]
    for num_buckets in [5, 10, 20]:
        feauture_name = "airport_code"
        hashed_feature = HashedFeature(feauture_name, num_buckets)
        hashed_results = {}
        for value in sample_values:
            bucket = hashed_feature.hash(value)
            hashed_results[value] = bucket
        print(f"Hashed Feature Results with {num_buckets=}:")
        # collision check
        bucket_counts = {}
        for value, bucket in hashed_results.items():
            print(f"Value: {value}, Bucket: {bucket}")
            if bucket not in bucket_counts:
                bucket_counts[bucket] = 0
            bucket_counts[bucket] += 1
        print("\nBucket Collision Counts:")
        for bucket, count in bucket_counts.items():
            print(f"Bucket: {bucket}, Count: {count}")
        print("-" * 30)
        
    
