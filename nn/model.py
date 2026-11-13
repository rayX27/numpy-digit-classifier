class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def params_and_grads(self):
        for layer in self.layers:
            for name in layer.params:
                yield layer.params, layer.grads, name