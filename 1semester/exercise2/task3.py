#!/usr/bin/env python3

def sortList(listOfDics):
    return sorted(listOfDics, key=lambda dic: dic['color'])


if __name__ == "__main__":
    print(sortList([{'make': 'Nokia',
                     'model': 216,
                     'color': 'Black'},
                    {'make': 'Mi Max',
                     'model': '2',
                     'color': 'Gold'},
                    {'make': 'Samsung',
                     'model': 7,
                     'color': 'Blue'}]))
