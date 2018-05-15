import unittest


class BowlingGame(object):
    def __init__(self):
        self.throws = []
        self.score = 0

    def throw(self, pins):
        self.throws.append(pins)

    def calculate_score(self):
        ball = 0
        for frames in range(10):
            if (self.throws[ball] == 10):
                self.score += 10 + self.throws[ball + 1] + self.throws[ball + 2]
                ball += 1
            elif (self.throws[ball] + self.throws[ball + 1] == 10):
                self.score += 10 + self.throws[ball + 2]
                ball += 2
            else:
                self.score += self.throws[ball] + self.throws[ball + 1]
                ball += 2


class BowlingGameTests(unittest.TestCase):
    def throw_many(self, game, number_of_times, pins):
        for _ in range(number_of_times):
            game.throw(pins)

    def test_all_gutters(self):
        game = BowlingGame()
        self.throw_many(game, 20, 0)
        game.calculate_score()
        self.assertEquals(game.score, 0)

    def test_all_ones(self):
        game = BowlingGame()
        self.throw_many(game, 20, 1)
        game.calculate_score()
        self.assertEquals(game.score, 20)

    def test_different_throws(self):
        game = BowlingGame()
        game.throw(6)
        game.throw(0)
        game.throw(7)
        game.throw(0)
        game.throw(2)
        self.throw_many(game, 15, 0)
        game.calculate_score()
        self.assertEquals(game.score, 15)

    def test_for_spare(self):
        game = BowlingGame()
        game.throw(4)
        game.throw(6)
        game.throw(7)
        game.throw(0)
        self.throw_many(game, 16, 0)
        game.calculate_score()
        self.assertEquals(game.score, 24)

    def test_for_strike(self):
        game = BowlingGame()
        game.throw(10)
        game.throw(4)
        game.throw(2)
        self.throw_many(game, 17, 0)
        game.calculate_score()
        self.assertEquals(game.score, 22)

    def test_perfect_game(self):
        game = BowlingGame()
        self.throw_many(game, 12, 10)
        self.throw_many(game, 8, 0)
        game.calculate_score()
        self.assertEquals(game.score, 300)


if __name__ == '__main__':
    unittest.main()
