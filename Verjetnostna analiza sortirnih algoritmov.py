import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# linear insertion sort
def linear_insertion_sort(shuffled_list):
    # we don't need to sort one element / empty list
    if len(shuffled_list) < 2:
        return 0

    return_list = [shuffled_list[0]]
    counter = 0

    for i in shuffled_list[1:]:
        found_insert_point = False
        for j in range(0, len(return_list)):
            counter += 1
            if i < return_list[j]:
                return_list.insert(j, i)
                found_insert_point = True
                break

        if not found_insert_point:
            return_list.append(i)

    return counter, return_list


# binary insertion sort
def binary_search(shuffled_list, val):
    counter = 0

    counter += 1
    if val < shuffled_list[0]:
        return counter, 0

    counter += 1
    if shuffled_list[-1] < val:
        return counter, len(shuffled_list)

    left = 0
    right = len(shuffled_list)

    while left < right:
        mid = math.ceil((left + right) / 2)
        counter += 1
        if shuffled_list[mid] < val:
            left = mid
        else:
            right = mid

        if right - left == 1:
            break

    return counter, right


def binary_insertion_sort(arr):
    return_list = [arr[0]]
    counter = 0
    for i in range(1, len(arr)):
        val = arr[i]
        noc_ip = binary_search(return_list, val)
        counter += noc_ip[0]
        return_list.insert(noc_ip[1], val)
    return counter, return_list


# heap sort
# To heapify subtree rooted at index i.
def heapify(arr, size_of_heap, i):
    counter = 0
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is greater than root
    if l < size_of_heap and arr[i] < arr[l]:
        counter += 1
        largest = l

    # See if right child of root exists and is greater than root
    if r < size_of_heap and arr[largest] < arr[r]:
        counter += 1
        largest = r

    # Change root, if needed
    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])  # swap

        # Heapify the root.
        counter += heapify(arr, size_of_heap, largest)

    return counter


# The main function to sort an array of given size
def heap_sort(arr):
    counter = 0
    length = len(arr)

    # Build a maxheap.
    # Since last parent will be at ((n//2)-1) we can start at that location.
    for i in range(length // 2 - 1, -1, -1):
        counter += heapify(arr, length, i)

    # One by one extract elements
    for i in range(length - 1, 0, -1):
        (arr[i], arr[0]) = (arr[0], arr[i])  # swap
        counter += heapify(arr, i, 0)

    return counter, arr


# merge sort
def merge_sort(shuffled_list):
    counter = 0

    if len(shuffled_list) > 1:

        # Create sub_array2 ← A[start...mid] and sub_array2 ← A[mid+1...end]
        mid = len(shuffled_list) // 2
        sub_list1 = shuffled_list[:mid]
        sub_list2 = shuffled_list[mid:]

        # Sort the two halves
        counter += merge_sort(sub_list1)[0]
        counter += merge_sort(sub_list2)[0]

        # Initial values for pointers that we use to keep track of where we are in each array
        i = j = k = 0

        # Until we reach the end of either start or end, pick larger among
        # elements start and end and place them in the correct position in the sorted array
        while i < len(sub_list1) and j < len(sub_list2):
            counter += 1
            if sub_list1[i] < sub_list2[j]:
                shuffled_list[k] = sub_list1[i]
                i += 1
            else:
                shuffled_list[k] = sub_list2[j]
                j += 1
            k += 1

        # When all elements are traversed in either arr1 or arr2,
        # pick up the remaining elements and put in sorted array
        while i < len(sub_list1):
            shuffled_list[k] = sub_list1[i]
            i += 1
            k += 1

        while j < len(sub_list2):
            shuffled_list[k] = sub_list2[j]
            j += 1
            k += 1
    return counter, shuffled_list


# quick sort
def quick_sort(counter_and_list):
    counter = counter_and_list[0]
    if len(counter_and_list[1]) <= 1:
        return counter, counter_and_list[1]
    else:
        pivot = counter_and_list[1][0]
        left = []
        right = []

        for x in counter_and_list[1][1:]:
            counter += 1
            if x < pivot:
                left.append(x)
            else:
                right.append(x)

        qs_left = quick_sort((0, left))
        qs_right = quick_sort((0, right))

        return_list = qs_left[1] + [pivot] + qs_right[1]

        counter += qs_left[0]
        counter += qs_right[0]

        return counter, return_list


n = 1000
print("n =", n)
list_1_to_n = [i for i in range(1, n + 1)]

results = []

number_of_sorts = 1000

for i in range(0, number_of_sorts):
    # randomizes the elements in the list
    random.shuffle(list_1_to_n)
    # uncomment the one you want to use and comment all other ones.
    #number_of_comparisons = linear_insertion_sort(list_1_to_n)
    number_of_comparisons = binary_insertion_sort(list_1_to_n)
    #number_of_comparisons = heap_sort(list_1_to_n)
    #number_of_comparisons = merge_sort(list_1_to_n)
    #number_of_comparisons = quick_sort((0, list_1_to_n))
    results.append(number_of_comparisons[0])

# Calculating the width of the intervals using the Freedman–Diaconis rule
q_34 = np.quantile(results, .75)
q_14 = np.quantile(results, .25)
l = 2*(q_34 - q_14)/len(results)**(1./3)
print("l =", l)

bins = int((max(results) - min(results))//l)
plt.hist(results, density=True, bins=bins)  # density=False would make counts

mean = np.mean(results)
stand_deviation = np.std(results)
print("var =", np.var(results))

print("mean =", mean)
print("stand_deviation =", stand_deviation)

interval = np.linspace(min(results), max(results), len(results))
pdf = norm.pdf(interval, loc=mean, scale=stand_deviation)

plt.plot(interval, pdf, color='red', label="normalna gostota")
plt.legend()
plt.show()
