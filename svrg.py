import torch
from torch.autograd import Variable
import random

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.rand(N, D_in), requires_grad=False)
y = Variable(torch.rand(N, D_out), requires_grad=False)

w1 = Variable(torch.rand(D_in, H), requires_grad=True)
w2 = Variable(torch.rand(H, D_out), requires_grad=True)

mu = []

learning_rate = 1e-3
for t in xrange(10):
	w1_snap = Variable(w1.data, requires_grad=True)
	#w1_snap = Variable(w1_snap)
	w2_snap = Variable(w2.data, requires_grad=True)
	#w2_snap = Variable(w2_snap)
	a = Variable(torch.zeros(D_in, H))
	for it in xrange(10):
		random_int = torch.LongTensor([random.randint(0,N-1)])
		ix = torch.index_select(x,0,random_int)
		iy = torch.index_select(y,0,random_int)
		y_pred = ix.mm(w1).clamp(min=0).mm(w2)
		y_pred_snap = ix.mm(w1_snap).clamp(min=0).mm(w2_snap)
		loss = (y_pred - y).pow(2).sum()
		loss_snap = (y_pred_snap - y).pow(2).sum()
		#print(t, loss.data[0])

		loss.backward()
		loss_snap.backward()
		
		a = a + Variable(w1_snap.grad.data)
		w1.data -= learning_rate * (w1.grad.data - w1_snap.grad.data)
		w2.data -= learning_rate * (w2.grad.data - w2_snap.grad.data)
		
		mu.append(Variable(w1_snap.grad.data, requires_grad = False))
		w1.grad.data.zero_()
		w2.grad.data.zero_()

		w1_snap.grad.data.zero_()
		w2_snap.grad.data.zero_()
	print a
	a = a/10
	print a
	break
