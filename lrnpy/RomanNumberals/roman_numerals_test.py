import unittest


# 1000 = M
# 100  = C
# 50   = L
# 10   = X
# 5    = V
# 1    = I

def convert(number):
    roman = ''

    arabic_numerals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    roman_numerals = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']

    for i in range(len(arabic_numerals)):
        while (number >= arabic_numerals[i]):
            roman += roman_numerals[i]
            number -= arabic_numerals[i]

    return roman


class RomanNumeralsTest(unittest.TestCase):
    def test_for_1(self):
        self.assertEqual(convert(1), 'I')

    def test_for_2(self):
        self.assertEqual(convert(2), 'II')

    def test_for_3(self):
        self.assertEqual(convert(3), 'III')

    def test_for_4(self):
        self.assertEqual(convert(4), 'IV')

    def test_for_5(self):
        self.assertEqual(convert(5), 'V')

    def test_for_9(self):
        self.assertEqual(convert(9), 'IX')

    def test_for_10(self):
        self.assertEqual(convert(10), 'X')

    def test_for_20(self):
        self.assertEqual(convert(20), 'XX')

    def test_for_22(self):
        self.assertEqual(convert(22), 'XXII')

    def test_for_27(self):
        self.assertEqual(convert(27), 'XXVII')

    def test_for_40(self):
        self.assertEqual(convert(40), 'XL')

    def test_for_42(self):
        self.assertEqual(convert(42), 'XLII')

    def test_for_44(self):
        self.assertEqual(convert(44), 'XLIV')

    def test_for_46(self):
        self.assertEqual(convert(46), 'XLVI')

    def test_for_50(self):
        self.assertEqual(convert(50), 'L')

    def test_for_90(self):
        self.assertEqual(convert(90), 'XC')

    def test_for_99(self):
        self.assertEqual(convert(99), 'XCIX')

    def test_for_100(self):
        self.assertEqual(convert(100), 'C')

    def test_for_400(self):
        self.assertEqual(convert(400), 'CD')

    def test_for_500(self):
        self.assertEqual(convert(500), 'D')

    def test_for_900(self):
        self.assertEqual(convert(900), 'CM')

    def test_for_1000(self):
        self.assertEqual(convert(1000), 'M')

    def test_for_2018(self):
        self.assertEqual(convert(2018), 'MMXVIII')




if __name__ == '__main__':
    unittest.main()
