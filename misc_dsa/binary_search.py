# implimentation of basic binary search and its variants 
def binary_search(x, ls):
    """
    Returns the index of x in ls, -1 if not found
    """
    l, r = 0, len(ls) - 1
    while l <= r:
        mid = (l + r)//2 
        if ls[mid] == x:
            return mid
        if ls[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return -1
    

def first_instance(x, ls):
    """
    Returns the index of first instance of x in ls, -1 if not found
    """
    l, r = 0, len(ls)-1
    first_instance = -1
    while l <= r:
        mid = (l + r)//2 
        if ls[mid] == x:
            first_instance = mid
            r = mid - 1
        elif ls[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return first_instance

def last_instance(x, ls):
    """
    Returns the index of the last instance of x in ls
    """
    l, r = 0, len(ls) - 1
    last = -1
    while l <= r:
        mid = (l + r)//2
        if ls[mid] == x:
            last = mid
            l = mid + 1
        elif ls[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return last

def nearest_smallest_element(x, ls):
    """
    Returns the nearest smallest element to x
    """
    l, r = 0, len(ls) - 1
    near = -1
    while l <= r:
        mid = (l + r)//2
        if ls[mid] == x:
            return mid
        if ls[mid] < x:
            near = mid
            l = mid + 1
        else:
            r = mid - 1
    return near

def nearest_largest_element(x, ls):
    """
    Returns the nearest largest element to x
    """
    l, r = 0, len(ls) - 1
    near = -1
    while l <= r:
        mid = (l + r)//2
        if ls[mid] == x:
            return mid
        if ls[mid] < x:
            l = mid + 1
        else:
            near = mid
            r = mid - 1
    return near

def search_in_rotated_sorted_array(x, ls):
    """
    Returns the index of element in rotated sorted array
    """
    l, r = 0, len(ls) - 1
    while l <= r:
        mid = (l + r)//2
        if ls[mid] == x:
            return mid

        # check if the left half is sorted
        if ls[l] <= ls[mid]:
            if ls[l] <= x < ls[mid]:
                r=mid-1
            else:
                l=mid+1
        else:
            if ls[mid] < x <= ls[r]:
                l=mid+1
            else:
                r=mid-1   
    return -1

def find_peak(ls):
    """
    find the index of peak of array, the array is inc and then decreasing
    """
    if not ls:
        return None
    if len(ls) == 1:
        return 0
    
    l, r = 0, len(ls) - 1

    while l < r:
        mid = (l + r)//2
        if ls[mid] > ls[mid + 1]:
            r = mid
        else: 
            l = mid + 1
    return l
        
 
ls = [1, 5, 6, 6, 8, 9, 10, 12, 15]

print(nearest_largest_element(11, ls))
print(nearest_smallest_element(7, ls))    
print(last_instance(6, ls))
print(first_instance(6, ls))

