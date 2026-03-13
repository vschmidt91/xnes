from leitwerk import Optimizer, Parameter

opt = Optimizer({"x": Parameter(), "y": Parameter()}, minimize=True)

def f(x, y):
    return (x - 1)**2 + (y - 1)**2

for _ in range(100):
    trial, params = opt.ask()
    opt.tell(trial, f(**params))

print(opt.ask_best())
# {'x': 1.007115753775713, 'y': 0.9922700335131514}