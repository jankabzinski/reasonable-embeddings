import numpy as np
import pandas as pd
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics

from src.simplefact import *
from src.simplefact.syntax import *
from src.utils import *
from src.vis import *

def core_mod(expr):
	if isinstance(expr, int) or expr == TOP or expr == BOT:
		return expr
	if expr[0] == NOT:
		inner = expr[1]
		if inner == TOP:
			return BOT
		if inner == BOT:
			return TOP
		if isinstance(inner, tuple):
			if inner[0] == OR:
				return AND, core_mod((NOT, inner[1])), core_mod((NOT, inner[2]))
			if inner[0] == ALL:
				return ANY, inner[1], core_mod((NOT, inner[2]))
		return NOT, core_mod(inner)
	if expr[0] == OR:
		assert len(expr) == 3
		return NOT, (AND, core_mod((NOT, expr[1])), core_mod((NOT, expr[2])))
	if expr[0] == AND:
		assert len(expr) == 3
		return expr[0], core_mod(expr[1]), core_mod(expr[2])
	if expr[0] == ALL:
		return NOT, (ANY, expr[1], core_mod((NOT, expr[2])))
	if expr[0] == ANY:
		return expr[0], expr[1], core_mod(expr[2])
	if expr[0] == SUB:
		return expr[0], core_mod(expr[1]), core_mod(expr[2])
	if expr[0] == DIS:
		assert len(expr) == 3
		return SUB, (AND, core_mod(expr[1]), core_mod(expr[2])), BOT
	assert False, f'bad expression {expr}'

def im_mod(c, d):
	cxd = T.outer(c, d).view(-1)
	return T.cat((c, d, cxd))

class ModifiedNeuralReasoner(nn.Module):
	def __init__(self, head=None, embs=None, emb_size=None, onto=None, hidden_size=None, hidden_count=1, seed=None):
		super().__init__()
		with T.random.fork_rng(enabled=seed is not None):
			if seed is not None: T.random.manual_seed(seed)

			self.head = head
			self.embs = embs

			if embs is not None:
				emb_size = self.embs.emb_size
			elif head is not None:
				emb_size = self.head.emb_size
			else:
				assert emb_size is not None

			if embs is None:
				assert onto is not None
				self.embs = ModifiedEmbeddingLayer.from_onto(onto, emb_size=emb_size)

			if head is None:
				self.head = ModifiedReasonerHead(emb_size=emb_size, hidden_size=hidden_size, hidden_count=hidden_count)

			assert self.head.emb_size == self.embs.emb_size

	def encode(self, expr):
		return self.head.encode(core_mod(expr), self.embs).detach().numpy()

	def check(self, axiom):
		return T.sigmoid(self.head.classify_batch([core_mod(axiom)], [self.embs]))[0].item()

	def check_sub(self, c, d):
		return self.check((SUB, c, d))

class ModifiedEmbeddingLayer(nn.Module):
	def __init__(self, *, emb_size, n_concepts, n_roles):
		super().__init__()
		self.n_concepts = n_concepts
		self.n_roles = n_roles
		self.emb_size = emb_size
		self.concepts = nn.Parameter(T.zeros((n_concepts, emb_size)))
		self.roles = nn.ModuleList([nn.Linear(emb_size, emb_size) for _ in range(n_roles)])
		nn.init.xavier_normal_(self.concepts)
		
	@classmethod
	def from_onto(cls, onto, *args, **kwargs):
		return cls(n_concepts=onto.n_concepts, n_roles=onto.n_roles, *args, **kwargs)

	@classmethod
	def from_ontos(cls, ontos, *args, **kwargs):
		return [cls.from_onto(onto, *args, **kwargs) for onto in ontos]

class InvolutiveLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.weight = nn.Parameter(T.Tensor(out_features, in_features))
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

	def forward(self, input):
		return F.linear(input, self.weight)

	# def involutive_loss(self, outputs):
	# 	identity = T.eye(self.weight.size(0), device=self.weight.device)
	# 	W_squared = T.matmul(self.weight, self.weight)
	# 	identity_loss = F.mse_loss(W_squared, identity) 
	# 	return  identity_loss

class ModifiedReasonerHead(nn.Module):
	def __init__(self, *, emb_size, hidden_size, hidden_count=1):
		super().__init__()
		self.hidden_size, self.emb_size = hidden_size, emb_size
		self.bot_concept = nn.Parameter(T.zeros((1, emb_size)))
		self.top_concept = nn.Parameter(T.zeros((1, emb_size)))
		self.not_nn = InvolutiveLinear(emb_size, emb_size)
		self.and_nn = nn.Linear(2*emb_size + emb_size**2, emb_size)

		sub_nn = [nn.Linear(2*emb_size + emb_size**2, hidden_size)]
		for _ in range(hidden_count - 1):
			sub_nn.append(nn.ELU())
			sub_nn.append(nn.Linear(hidden_size, hidden_size))
		sub_nn.append(nn.ELU())
		sub_nn.append(nn.Linear(hidden_size, 1))
		self.sub_nn = nn.Sequential(*sub_nn)

		self.rvnn_act = lambda x: x

		nn.init.xavier_normal_(self.bot_concept)
		nn.init.xavier_normal_(self.top_concept)
			
	def encode(self, axiom, embeddings):
		not_nn_outputs = []
		and_inputs = []
		def rec(expr):
			if expr == TOP:
				return self.rvnn_act(self.top_concept[0])
			elif expr == BOT:
				return self.rvnn_act(self.bot_concept[0])
			elif isinstance(expr, int):
				return self.rvnn_act(embeddings.concepts[expr])
			elif expr[0] == AND:
				c = rec(expr[1])
				d = rec(expr[2])
				and_inputs.append(c)
				and_inputs.append(d)
				return self.rvnn_act(self.and_nn(im_mod(c, d)))
			elif expr[0] == NOT:
				c = rec(expr[1])
				not_nn_out = self.rvnn_act(self.not_nn(c))
				not_nn_outputs.append([c, not_nn_out])
				return not_nn_out
			elif expr[0] == ANY:
				c = rec(expr[2])
				r = embeddings.roles[expr[1]]
				return self.rvnn_act(r(c))
			elif expr[0] == SUB:
				c = rec(expr[1])
				d = rec(expr[2])
				return self.sub_nn(im_mod(c, d))
			else:
				assert False, f'Unsupported expression {expr}. Did you convert it to core form?'
		return rec(axiom), not_nn_outputs, and_inputs
	
	def not_loss(self, output):
		orig = output[0]
		not_out = output[1]
		not_twice = self.not_nn(not_out)
		loss_recover = F.mse_loss(not_twice, orig)
		return loss_recover

	def and_loss(self, outputs):
		outputs = [item for sublist in outputs if sublist for item in sublist]

		loss = 0.0
		for output in outputs:
			loss += F.mse_loss(output, self.and_nn(im_mod(output, output))).item() #* 3
			# + F.mse_loss(self.bot_concept[0], self.and_nn(im_mod(self.bot_concept[0], output))) + 
			# F.mse_loss(output, self.and_nn(im_mod(self.top_concept[0], output)))).item()

		return T.tensor(loss/len(outputs), requires_grad = True)

	def classify_batch(self, axioms, embeddings):
		encoded_outputs = [self.encode(axiom, emb) for axiom, emb in zip(axioms, embeddings)]
		final_outputs = T.vstack([out[0] for out in encoded_outputs])
		not_nn_embeddings = [out[1] for out in encoded_outputs]
		and_inputs = [out[2] for out in encoded_outputs]
		return final_outputs, not_nn_embeddings, and_inputs
	
	def classify(self, axiom, emb):
		return self.classify_batch([axiom], [emb])[0].item()

def batch_stats_mod(Y, y, **other):
	K = np.array(Y) > 0.5
	roc_auc = metrics.roc_auc_score(y, Y)
	pr_auc = metrics.average_precision_score(y, Y)
	acc = metrics.accuracy_score(y, K)
	f1 = metrics.f1_score(y, K)
	prec = metrics.precision_score(y, K, zero_division=0)
	recall = metrics.recall_score(y, K)
	return dict(acc=acc, f1=f1, prec=prec, recall=recall, roc_auc=roc_auc, pr_auc=pr_auc, **other)


def eval_batch_mod(reasoner, encoders, X, y, onto_idx, indices=None, *, backward=False, detach=True, not_nn_loss_weight=25, and_nn_loss_weight=12):
	if indices is None: indices = list(range(len(X)))
	emb = [encoders[onto_idx[i]] for i in indices]
	X_ = [core_mod(X[i]) for i in indices]
	y_ = T.tensor([float(y[i]) for i in indices]).unsqueeze(1)
	Y_, not_outputs, and_inputs = reasoner.classify_batch(X_, emb)
	main_loss = F.binary_cross_entropy_with_logits(Y_, y_, reduction='mean')
	
	# if backward:
	# 	main_loss.backward(retain_graph=True)
	# 	reasoner.not_nn.zero_grad()

	not_losses = []
	for outs in not_outputs:
		if outs:
			not_losses.append(T.stack([reasoner.not_loss(ne) for ne in outs]).mean())
	
	if not_losses:
		not_loss = T.stack(not_losses).mean()
	else:
		not_loss = T.tensor(0.0, device=Y_.device, requires_grad=False)
	
	and_loss = reasoner.and_loss(and_inputs)

	loss = main_loss + not_loss * not_nn_loss_weight + and_loss * and_nn_loss_weight
	if backward:
		loss.backward()

	Y_ = T.sigmoid(Y_)
	if detach:
		loss = loss.item()
		y_ = y_.detach().numpy().reshape(-1)
		Y_ = Y_.detach().numpy().reshape(-1)
	
	return loss, list(y_), list(Y_)

def train_mod(data_tr, data_vl, reasoner, encoders, *, epoch_count=15, batch_size=32, logger=None, validate=True,
			  optimizer=T.optim.AdamW, lr_reasoner=0.0001, lr_encoder=0.0002, freeze_reasoner=False, run_name='train', not_nn_loss_weight=25, and_nn_loss_weight=12):
	idx_tr, X_tr, y_tr = data_tr
	idx_vl, X_vl, y_vl = data_vl if data_vl is not None else data_tr
	if logger is None:
		logger = TrainingLogger(validate=validate, metrics=batch_stats_mod)

	optimizers = []
	for encoder in encoders:
		optimizers.append(optimizer(encoder.parameters(), lr=lr_encoder))

	if freeze_reasoner:
		freeze(reasoner)
	else:
		optimizers.append(optimizer(reasoner.parameters(), lr=lr_reasoner))

	logger.begin_run(epoch_count=epoch_count, run=run_name)
	for epoch_idx in range(epoch_count + 1):
		# Training
		batches = minibatches(T.randperm(len(X_tr)), batch_size)
		logger.begin_epoch(batch_count=len(batches))
		for idxs in batches:
			for optim in optimizers:
				optim.zero_grad()
			loss, yb, Yb = eval_batch_mod(reasoner, encoders, X_tr, y_tr, idx_tr, idxs, backward=epoch_idx > 0, not_nn_loss_weight=not_nn_loss_weight, and_nn_loss_weight=and_nn_loss_weight)
			for optim in optimizers:
				optim.step()
			logger.step(loss)

		# Validation
		if validate:
			with T.no_grad():
				val_loss, yb, Yb = eval_batch_mod(reasoner, encoders, X_vl, y_vl, idx_vl, not_nn_loss_weight=not_nn_loss_weight)
				logger.step_validate(val_loss, yb, Yb, idx_vl)

		logger.end_epoch()

	if freeze_reasoner:
		unfreeze(reasoner)

	return logger
