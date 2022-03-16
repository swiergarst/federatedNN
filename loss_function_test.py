from numpy import double
import torch 
import torch.nn as nn



def custom_loss(output, target,x, y, z, rho):
    fx = nn.CrossEntropyLoss()
    #fx_loss = fx(torch.reshape(output, (-1, 1)), target)

    fx_loss = fx(output, target)

    y_res = torch.reshape(y.weight, (-1,))
    sub = torch.reshape(torch.sub(x.weight, z.weight), (-1,))
    first_order = torch.dot(y_res, sub)
    second_order = (rho/2) * torch.dot(sub, sub)
    #print(fx_loss)
    #print(first_order.shape)
    #print(second_order.shape)
    return(fx_loss + first_order + second_order)


input = torch.randn((1,2), dtype=torch.double, requires_grad=True)
#target = torch.randint(low=0, high=1, size=(2,))
target = torch.empty(1, dtype=torch.long).random_(2)
x = nn.Linear(2,2).double()
y = nn.Linear(2,2).double()
z = nn.Linear(2,2).double()

output = x(input)

#torch.set_default_tensor_type(torch.DoubleTensor)

loss = custom_loss(output, target, x,y,z, 0.5)
#print(loss)
loss.backward()
print(x.weight.grad)