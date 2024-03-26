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

print(y.transpose(1, 2).contiguous())
y = y.repeat(4, 1, 1)
print(y.shape)

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
