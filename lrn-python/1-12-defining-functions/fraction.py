#!/usr/bin/env python

"""Build a fraction class

2016-12-06
http://interactivepython.org/runestone/static/pythonds/Introduction/ObjectOrientedProgramminginPythonDefiningClasses.html

"""


def gcd(m, n):
    """Find the greatest common denominator"""
    while m % n != 0:
        old_m = m
        old_n = n

        m = old_n
        n = old_m % old_n
    return n


class Fraction:
    """The Fraction class"""

    def __init__(self, top, bottom):
        'Constructor'
        self.num = top
        self.den = bottom

    def __str__(self):
        return '{}/{}'.format(str(self.num), str(self.den))

    def show(self):
        return '{}/{}'.format(str(self.num), str(self.den))

    def __repr__(self):
        return '{}/{}'.format(str(self.num), str(self.den))

    def __add__(self, other):
        new_num = self.num * other.den + self.den * other.num
        new_den = self.den * other.den
        common = gcd(new_num, new_den)

        return Fraction(new_num / common, new_den / common)

    def __eq__(self, other):
        first_num = self.num * other.den
        second_num = other.num * self.den

        return first_num == second_num


def main():
    myfraction = Fraction(1, 4)
    print 'myfraction = ', myfraction

    other_frac = Fraction(1, 2)
    print 'other_frac = ', other_frac

    sum_frac = myfraction + other_frac
    print 'sum_frac = ', sum_frac

    print 'gcd (20, 10) = ', gcd(20, 10)

    res = Fraction(1, 4) == Fraction(3, 6)
    print 'Fraction (2, 4) == Fraction (3, 6) : ', res

    x = Fraction(1, 2)
    y = Fraction(2, 3)
    z = Fraction(2, 4)
    m = Fraction(3, 6)

    print (x + y)
    print (x == y)
    print (x == z)
    print m == z


if __name__ == '__main__':
    main()
