#!/usr/bin/env python3

def sortList(listOfTuples):
    return sorted(listOfTuples, key=lambda tup: tup[1])


if __name__ == "__main__":
    print(sortList([(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]))
