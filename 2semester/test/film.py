#!/usr/bin/env python3

class Film:

    def __init__(
            self,
            title,
            director,
            actors,
            genre,
            country,
            language,
            rating,
            popularity) -> None:
        self._title = title
        self._director = director
        self._actors = actors
        self._genre = genre
        self._country = country
        self._language = language
        self._rating = rating
        self._popularity = popularity

    def __repr__(self) -> str:
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def director(self):
        return self._director

    @director.setter
    def director(self, director):
        self._director = director

    @property
    def actors(self):
        return self._actors

    @actors.setter
    def actors(self, actors):
        self._actors = actors

    @property
    def genre(self):
        return self._genre

    @genre.setter
    def genre(self, genre):
        self._genre = genre

    @property
    def country(self):
        return self._country

    @country.setter
    def country(self, country):
        self._country = country

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, language):
        self._language = language

    @property
    def rating(self):
        return self._rating

    @rating.setter
    def rating(self, rating):
        self._rating = rating

    @property
    def popularity(self):
        return self._popularity

    @popularity.setter
    def popularity(self, popularity):
        self._popularity = popularity
