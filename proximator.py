"""
proximator for fista/admm
"""

class proximator():

    def __init__(self, net, chns=2, iteration=3):
        self.net       = net
        self.chns      = chns
        self.iteration = iteration

    def forward(self, x_k0):
        x_init = x_k0
        for _ in range(self.iteration):
            gradient = self.net.model(x_k0) - x_init
            x_k1 = x_k0 - gradient
            x_k0 = x_k1
        return x_k0