from abc import ABCMeta, abstractmethod

class Attacker(metaclass=ABCMeta):
    @abstractmethod
    def attack(self, inputs, targets):
        raise NotImplementedError
