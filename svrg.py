from __future__ import division
import torch
from torch.autograd import Variable
import random

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.rand(N, D_in), requires_grad=False)
y = Variable(torch.rand(N, D_out), requires_grad=False)

w1 = Variable(torch.rand(D_in, H), requires_grad=True)
w2 = Variable(torch.rand(H, D_out), requires_grad=True)

mu_1 = Variable(torch.zeros(D_in,H))
mu_2 = Variable(torch.zeros(H, D_out))

w1_snap_stack = [Variable(w1.data, requires_grad=True)]
w2_snap_stack = [Variable(w2.data, requires_grad=True)]

learning_rate = 1e-4
for t in xrange(10):
	w1_snap = w1_snap_stack[-1] #Variable(w1.data, requires_grad=True)
	#w1_snap = Variable(w1_snap)
	w2_snap = w2_snap_stack[-1] #Variable(w2.data, requires_grad=True)
	#w2_snap = Variable(w2_snap)
	a = Variable(torch.zeros(D_in, H))
	b = Variable(torch.zeros(H,D_out))
	for it in xrange(5):
		random_int = torch.LongTensor([random.randint(0,N-1)])
		ix = torch.index_select(x,0,random_int)
		iy = torch.index_select(y,0,random_int)
		y_pred = ix.mm(w1).clamp(min=0).mm(w2)
		y_pred_snap = ix.mm(w1_snap).clamp(min=0).mm(w2_snap)
		loss = (y_pred - iy).pow(2).sum()
		loss_snap = (y_pred_snap - iy).pow(2).sum()
		#print(t, loss.data[0])

		loss.backward()
		loss_snap.backward()
		
		a = a + Variable(w1_snap.grad.data)
		b = b + Variable(w2_snap.grad.data)
		w1.data -= learning_rate * (w1.grad.data - w1_snap.grad.data + mu_1.data)
		w2.data -= learning_rate * (w2.grad.data - w2_snap.grad.data + mu_2.data)
		
		w1.grad.data.zero_()
		w2.grad.data.zero_()

		w1_snap.grad.data.zero_()
		w2_snap.grad.data.zero_()
	#print w1.data, w2.data
	mu_1 = a/5
	mu_2 = b/5
	w1_snap_stack.append(Variable(w1.data, requires_grad = True))
	w2_snap_stack.append(Variable(w2.data, requires_grad = True))

#print "vanilla nn"
'''
for it in xrange(50):
	random_int = torch.LongTensor([random.randint(0,N-1)])
	ix = torch.index_select(x,0,random_int)
	iy = torch.index_select(y,0,random_int)
	y_pred = ix.mm(w1).clamp(min=0).mm(w2)
	loss = (y_pred - iy).pow(2).sum()
	print loss.data[0]
	loss.backward()
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data

	w1.grad.data.zero_()
	w2.grad.data.zero_()
'''
