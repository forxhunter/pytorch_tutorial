import torch
from torchviz import make_dot
x = torch.ones (5)
y = torch.zeros(3)
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3, requires_grad=True)

y_pred = torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred,y)
#二元交叉熵
graph = make_dot(y_pred, params={'w': w, 'b': b})
# 保存在当下路径下
# graph.view()
graph.render(filename="./inmg/compu_map_tz",view=True,format="png")
