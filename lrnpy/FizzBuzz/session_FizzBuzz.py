def fizzbuzz(number):

    # if (number % 15 == 0):
    #     return "FizzBuzz"
    # if (number % 3 == 0):
    #     return "Fizz"
    # if (number % 5 == 0):
    #     return "Buzz"
    # return str(number)

    # # build string
    # result = ''
    # if (number % 3 == 0):
    #     result += 'Fizz'
    # if (number % 5 == 0):
    #     result += 'Buzz'
    # return result or str(number)

    if divisible_by_3(number) and divisible_by_5(number):
        return "FizzBuzz"
    if divisible_by_3(number):
        return "Fizz"
    if divisible_by_5(number):
        return "Buzz"
    return str(number)

def divisible_by_3(number):
    return (number % 3 == 0)

def divisible_by_5(number):
    return (number % 5 == 0)
