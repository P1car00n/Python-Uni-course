def count_vowels (string):
    vowels = [vowel for vowel in string if vowel.lower() in ('a','o','u','e','i')]
    return ("Number of vowels: " + str(len(vowels)))
    # print(vowels)

print(count_vowels('Test string askdjajii9d92dj awdj9 oooOOOO'))