#!/usr/bin/env python3

import string


def count_words(sentence):
    words = sentence.translate(str.maketrans(
        '', '', string.punctuation)).split()
    count = {word: words.count(word) for word in words}
    output = ''
    for key, value in count.items():
        output = output + '{} = {}, '.format(key, value)
    return output[:-2]  # account for the trailing ', '


if __name__ == "__main__":
    print(count_words('Tom, eats, and eats.'))
