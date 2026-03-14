from leitwerk import Optimizer, Parameter

opt = Optimizer({"a": Parameter(), "b": Parameter()}, minimize=True)

for _ in range(500):
    x = opt.ask()
    opt.tell((x["a"] - 1) ** 2 + (x["b"] - 2) ** 2)

print(opt.mean)