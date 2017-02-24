# http://interactivepython.org/runestone/static/pythonds/Introduction/ObjectOrientedProgramminginPythonDefiningClasses.html

class LogicGate:
    def __init__(self, n):
        self.label = n
        self.output = None

    def getLabel(self):
        return self.label

    def getOutput(self):
        self.output = self.performGateLogic()
        return self.output

class BinaryGate(LogicGate):

    def __init__(self, n):
        LogicGate.__init__(self, n)

        self.pinA = None
        self.pinB = None

    def getPinA(self):
        if self.pinA == None:
            input_str = 'Enter Pin A input for gate {} -->'
            return int(input(input_str.format(self.getLabel())))
        else:
            return self.pinA.getFrom().getOutput()

    def getPinB(self):
        if self.pinB == None:
            input_str = 'Enter Pin B input for gate {} -->'
            return int(input(input_str.format(self.getLabel())))
        else:
            return self.pinB.getFrom().getOutput()

    def setNextPin(self, source):
        if self.pinA == None:
            self.pinA = source
        else:
            if self.pinB == None:
                self.pinB = source
            else:
                raise RuntimeError('Cannont Connect: NO EMPTY PINS on this gate')



class UnaryGate(LogicGate):

    def __init__(self, n):
        LogicGate.__init__(self, n)

        self.pin = None

    def getPin(self):
        if self.pin == None:
            input_str = 'Enter Pin input for gate {} -->'
            return int(input(input_str.format(self.getLabel())))
        else:
            return self.pin.getFrom().getOutput()

    def setNextPin(self, source):
        if self.pin == None:
            self.pin = source
        else:
            raise RuntimeError('Cannont Connect: NO EMPTY PINS on this gate')

class AndGate(BinaryGate):
    def __init__(self, n):
        BinaryGate.__init__(self, n)

    def performGateLogic(self):
        a = self.getPinA()
        b = self.getPinB()
        res = 1 if a == 1 and b == 1 else 0
        return res

class OrGate(BinaryGate):
    def __init__(self, n):
        BinaryGate.__init__(self, n)

    def performGateLogic(self):
        a = self.getPinA()
        b = self.getPinB()
        res = 1 if a == 1 or b == 1 else 0
        return res

class NotGate (UnaryGate):
    def __init__ (self, n):
        UnaryGate.__init__ (self, n)

    def performGateLogic (self):
        a = self.getPin ()
        res = 0 if a == 1 else 1
        return res 

class Connector:
    def __init__(self, fgate, tgate):
        self.fromgate = fgate
        self.togate = tgate
        tgate.setNextPin (self)

    def getFrom (self):
        return self.fromgate

    def getTo(self):
        return self.togate

def main():
    g1 = AndGate ('G1')
    g1.getOutput ()

if __name__ == '__main__':
    main()
