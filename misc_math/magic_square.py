# from itertools import permutations, combinations
from pprint import pprint


def check(permutation, n):
    matrix = [permutation[i:i+n] for i in range(0, len(permutation), n)]

    # check rows sum equal
    row_sum = sum(matrix[0])
    for row in matrix:
        if sum(row) != row_sum:
            return False
    
    # check columns sum equal
    for j in range(n):
        col_sum = sum(row[j] for row in matrix)
        if col_sum != row_sum:
            return False
    
    # check diagonals sum equal
    diag1_sum = sum(matrix[i][i] for i in range(n))
    diag2_sum = sum(matrix[i][n-i-1] for i in range(n))
    if diag1_sum != row_sum or diag2_sum != row_sum:
        return False
    
    return True

# def permutations(i, remaining, n, cur_str):
#     if i >= n:
#         return [cur_str]
#     res = []
#     for j, ele in enumerate(remaining):
#         res += (permutations(i+1, remaining[:j] + remaining[j+1:], n, cur_str+str(ele)))
#     return res


def magic_square(n):
    res = []
    
    # for i in range(n):
    #     res.append(i, )
    def permutations(i, path, n, cur_str):
        if i >= n:
            # print(cur_str)
            res.append(cur_str)
        for j in range(1, n+1):
            if j not in set(path):
                permutations(i+1, path + [j], n, cur_str+[j])
    permutations(0, [], n**2, [])
    op = []
    for permutation in res:
        if check(permutation, n):
            matrix = [permutation[i:i+n] for i in range(0, len(permutation), n)]
            op.append(matrix)

    return op

pprint(magic_square(3))

