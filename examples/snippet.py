from leitwerk import Optimizer, Parameter

def f(x1, x2):
    return (x1 - 1)**2 + (x2 - 1)**2  # minimum at (1, 1)

opt = Optimizer({"x1": Parameter(), "x2": Parameter()}, minimize=True)

for _ in range(100):
    x = opt.ask()
    opt.tell(f(**x))

print(opt.ask_best())