#!/usr/bin/env python

"""How long until a monkey can randomly type the string
'methinks it is like a weasel'

From http://interactivepython.org/runestone/static/pythonds/Introduction/DefiningFunctions.html
"""

import random
import string


def main():
    target_string = 'methinks it is like a weasel'
    target_string = 'bartev v'
    strlen = len(target_string)

    ctr = 0
    new_string = generateOne(strlen)
    best_string = new_string

    new_score = score (target_string, new_string)
    best_score = new_score

    print target_string

    while new_score < 0.95:
        if new_score > best_score:
            # print new_score, ' ', new_string
            best_score = new_score
            best_string = new_string

        if ctr % 1000000 == 0:
            print best_score, ' ', best_string, ' ', ctr/1000000
        elif ctr % 100000 == 0:
            print '.',

        new_string = generateOne(strlen)
        new_score = score (target_string, new_string)
        ctr += 1

def generateOne(strlen):
    """generate a random string that is `strlen` long
    """
    alphabet = string.ascii_lowercase + ' '
    res = ""
    for i in range(strlen):
        res = res + alphabet[random.randrange(27)]

    return res


def score(goal, teststring):
    """calc pct of chars that match target string
    """
    num_same = 0
    for i in range(len(goal)):
        if goal[i] == teststring[i]:
            num_same += 1
    return 1.0 * num_same / len(goal)


if __name__ == '__main__':
    main()
