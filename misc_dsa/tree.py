class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left 
        self.right = right


def LCA(node, p, q):
    if not node or node == p or node == q:
        return node
    
    left = LCA(node.left, p, q)
    right = LCA(node.right, p, q)

    if left and right:
        return node
    
    return left if left else right

