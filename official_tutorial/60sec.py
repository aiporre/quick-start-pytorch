from __future__ import print_function
import torch

# x = torch.empty(5, 3)
# print(x)
# x = torch.rand(5, 3)
# print(x)
# x = torch.zeros(10, 3, dtype=torch.long)
# print(x)
x = torch.tensor([5.5, 3])
print ("simple vector")
print(x)

# create tensors from exisitng tensors:
print('overide of values...')
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

# printing size
print('printing size of the matrix')
print(x.size())


# OPERATIONS
print('some operations...')
y = torch.rand(5, 3)
# one way
print('that is x plus y')
print(x + y)
# another way
print('another way to sum')
print(torch.add(x, y))
# addtion in situs
print('y is sum in place in place')
y.add_(x)
print(y)

# mutation in place
print('any operation terminated with _ means that is tit taking the same value')
x.t_()
print(x)

# resizing a vector
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
