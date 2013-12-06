class Abstract:

    def __init__(self, a):
        self.a = a


class Derived(Abstract):

    def __init__(self, a):
        Abstract.__init__(self, a)

    def Print(self):
        print "this is my a:", self.a # WORKS :D

if __name__ == "__main__":
    obj = Derived(3)
    obj.Print()
