from teenygrad.tensor import Tensor

rand = Tensor.randn(2, 3)
#print(rand.numpy())

arg = [[1, 2], [2, 2], [3, 4]]
t = tuple(slice(p[0], p[1], None) for p in arg)
#print(t)


t1 = Tensor.randn(2, 3)
t2 = Tensor.randn(2, 3)
result = t1 < t2
print(result.lazydata.toCPU())