#!/usr/bin/env python3

from film import Film

titles = ['Title' + str(x) for x in range(1, 11)]
directors = ['Director' + str(x) for x in range(1, 11)]
actors = ['Actor' + str(x) for x in range(1, 11)]
genres = ['Genre' + str(x) for x in range(1, 11)]
countries = [
    'Uganda',
    'Uganda',
    'Uganda',
    'Uganda',
    'Uganda',
    'Germany',
    'Germany',
    'Germany',
    'Germany',
    'Germany']
languages = [
    'Ugandian',
    'Ugandian',
    'Ugandian',
    'Ugandian',
    'Ugandian',
    'German',
    'German',
    'German',
    'German',
    'German']
languages = [
    'Ugandian',
    'Ugandian',
    'Ugandian',
    'Ugandian',
    'Ugandian',
    'German',
    'German',
    'German',
    'German',
    'German']
ratings = [4, 4, 5, 5, 1, 3, 5, 2, 2, 2]
popularity = [4, 4, 5, 5, 1, 3, 5, 2, 2, 2]

films = []

for i in range(0, 10):
    films.append(
        Film(
            titles[i],
            directors[i],
            actors,
            genres[i],
            countries[i],
            languages[i],
            ratings[i],
            popularity[i]))

print('Current films: ')
for i in films:
    print(i)


def check_dependency_country_rating(films):
    countries = []
    ratings = []
    for i in films:
        countries.append(i.country)
        ratings.append(i.rating)

    dependency = zip(countries, ratings)
    return dependency


def check_dependency_language_popularity(films):
    languages = []
    popularity = []
    for i in films:
        languages.append(i.language)
        popularity.append(i.rating)

    dependency = zip(languages, popularity)
    return dependency


print('Country to rating:')
for i in check_dependency_country_rating(films):
    print(i)

print('Language to popularity:')
for i in check_dependency_language_popularity(films):
    print(i)
