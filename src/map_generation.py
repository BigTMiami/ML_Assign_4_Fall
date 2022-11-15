from random import random


def gen_map(size, prob):
    lake_map = []
    for i in range(size):
        row_string = ""
        for j in range(size):
            if i == 0 and j == 0:
                tile = "S"
            elif i == size - 1 and j == size - 1:
                tile = "G"
            else:
                tile = "H" if random() < prob else "F"
            row_string += tile
        lake_map.append(row_string)
    return lake_map


large_map = gen_map(16, 0.02)
large_map
