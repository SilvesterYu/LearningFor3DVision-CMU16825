import torch

x = torch.Tensor([
[
    [1, 2, 3],
    [1, 2, 3]
],

[
    [2, 2, 2],
    [2, 2, 2]
],

[
    [1, 2, 1],
    [1, 2, 1]
],

[
    [4, 4, 4],
    [5, 5, 5]
]
])

print(torch.matmul(x, torch.transpose(x, 1, 2)))

y = torch.Tensor(
    [
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
    ]
    )

# print(y.transpose(1, 2).contiguous())
# y = y.repeat(4, 1, 1)
# print(y.shape)

# a = torch.rand(2, 3)
# print(a)
# b = torch.eye(a.size(1))
# c = a.unsqueeze(2).expand(*a.size(), a.size(1))
# d = c * b
# print(d)
# print(a)
# print(a.unsqueeze(2))
# print(c)
# print(b)
# print(d)

hey = torch.randn(5, 3)
mask = hey[:, -1] > 0
idx = torch.linspace(0, hey.shape[0]-1, hey.shape[0])
print(hey.shape[0])
print("hey", hey)
print("mask", mask)
print("idx", idx)
res = torch.masked_select(idx, mask)
res = res.to(torch.int64)
print("Res", res)
print(type(res))
print(res.type())

z_vals = torch.randn(1, 6)
print(z_vals)
sorted, indices = torch.sort(z_vals)
mask = sorted >= 0
idxs = torch.masked_select(indices, mask).to(torch.int64)
print(idxs)

m = torch.randn(3, 2, 4)
print("m)", m)
print(torch.sum(m, dim=0))


this = torch.randn(5, 2, 4, 1)
that = torch.randn(5, 1, 1, 3)
res = that * this * this
print(res)
print(res.shape)


this = torch.randn(5, 2, 3)
print(this)
that = torch.cumprod(this, dim = 0)
print(that)