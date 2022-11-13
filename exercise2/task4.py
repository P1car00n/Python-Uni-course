#!/usr/bin/env python3

def stringToList(listOfStrings):
    return list(map(list, listOfStrings))


if __name__ == "__main__":
    print(stringToList(
        ['This is a list of strings.', 'Dummy text', '33zz   z.,!!']))
