import torch


def generate_coeffs():
    a = torch.rand(size=()) * 10
    b = -10 + torch.rand(size=()) * 10
    c = -10 + torch.rand(size=()) * 10
    return a, b, c


def func(x, a, b, c):
    return x.pow(2) * a + x * b + c


# def find_min(a, b, c):
#     dtype = torch.float
#     device = torch.device("cpu")
#
#     x = torch.randint(-1, 1, (1,), dtype=dtype, device=device, requires_grad=True)
#     alpha = 1e-2
#     stop = 1e-8
#
#     while x.grad is None:
#         y = func(x=x, a=a, b=b, c=c)
#         y.backward()
#         update = x.data - alpha * x.grad.data
#         f_x = func(x=update, a=a, b=b, c=c)
#         if torch.abs(x - update) > stop:
#             x.grad = None
#         x.data = update
#
#     return x.data[0], func(x=x.data, a=a, b=b, c=c)

# def find_min(a, b, c):
#     dtype = torch.float
#     device = torch.device("cpu")
#
#     x = torch.randint(-1, 1, (1,), dtype=dtype, device=device, requires_grad=True)
#     alpha = 1e-2
#     last_grad = 100.
#     stop = 1e-7
#
#     while x.grad is None:
#         y = func(x=x, a=a, b=b, c=c)
#         y.backward()
#         update = x.data - alpha * x.grad.data
#         if torch.abs(last_grad - x.grad.data) > stop:
#             last_grad = x.grad
#             x.grad = None
#         x.data = update
#
#     return x.data[0], func(x=x.data, a=a, b=b, c=c)

def find_min(a, b, c):
    dtype = torch.float
    device = torch.device("cpu")

    x = torch.randint(-1, 1, (1,), dtype=dtype, device=device, requires_grad=True)
    alpha = 1e-2
    stop_precision = 1e-7

    while x.grad is None:
        y = func(x=x, a=a, b=b, c=c)
        y.backward()
        update = x.data - alpha * x.grad.data
        if torch.abs(x - update) > stop_precision:
            x.grad = None
        x.data = update

    return x.data[0], func(x=x.data, a=a, b=b, c=c)


ctr = 0
iters = 1
for i in range(iters):
    coeffs = generate_coeffs()
    expected = -coeffs[1] / (2. * coeffs[0])
    expected_y = coeffs[0] * (expected ** 2) + coeffs[1] * expected + coeffs[2]
    found_x, found_y = find_min(*coeffs)
    print(type(found_x))
    print(type(found_y))
    try:
        assert torch.abs(expected - found_x) <= 1e-3
    except AssertionError:
        ctr += 1
        print(expected)
        print(found_x)

print(100 - ((ctr / iters) * 100))
