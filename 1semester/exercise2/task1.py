#!/usr/bin/env python3

import string
import re


def count_words(sentence):
    words = re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)
    count = {word: words.count(word) for word in words if word not in string.punctuation}
    output = ''
    for key, value in count.items():
        output = output + '{} = {}, '.format(key, value)
    return output[:-2]  # account for the trailing ', '


if __name__ == "__main__":
    print(count_words(input('Word counter: ')))
