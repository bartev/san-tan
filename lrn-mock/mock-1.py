# https://docs.python.org/3/library/unittest.mock-examples.html

from unittest.mock import MagicMock, Mock
from unittest.mock import patch


class ProductionClass:
    """Documentation for ProductionClass

    """
    def method(self):
        self.something(1, 2, 3)

    def something(self, a, b, c):
        pass


real = ProductionClass()
real.something = MagicMock()
real.method()
real.something.assert_called_once_with(1, 2, 3)

real.something.assert_called_once_with(1, 2, 3, 4)

##########


class ProductionClass2:
    """Documentation for ProductionClass

    """
    def closer(self, something):
        something.close()


real = ProductionClass2()
mock = Mock()
real.closer(mock)
mock.close.assert_called_with()


##########

def some_function():
    instance = module.Foo()
    return instance.method()


with patch('module.Foo') as mock:
    instance = mock.return_value
    instance.method.return_value = 'the result'
    result = some_function()
    assert result == 'the result'
