import copy
from typing import Tuple

import numpy as np
from tqdm import tqdm

from ElNaive import find_nontrivial_proofs, ElNaive
from src.simplefact.syntax import *


def random_axiom(n_concepts: int, n_roles: int, rng: np.random.Generator):
	if n_roles > 0:
		axiom_type = rng.choice(4)
	else:
		axiom_type = rng.choice(2)
	if axiom_type == 0:
		# CR1
		c, d = rng.integers(0, n_concepts, 2)  # nor here
		return SUB, int(c), int(d)
	elif axiom_type == 1:
		# CR2
		c1, c2, d = rng.integers(0, n_concepts, 3)  # don't use TOP and BOT here, it makes no sense
		return SUB, (AND, int(c1), int(c2)), int(d)
	elif axiom_type == 2:
		# CR3
		c, d = rng.integers(0, n_concepts + 2, 2) - 2
		r = rng.integers(0, n_roles, 1)
		return SUB, int(c), (ANY, int(r), int(d))
	else:
		assert axiom_type == 3
		# CR4
		d, e = rng.integers(0, n_concepts + 2, 2) - 2
		r = rng.integers(0, n_roles, 1)
		return SUB, (ANY, int(r), int(d)), int(e)


def gen(n_concepts: int, n_roles: int, rng: np.random.Generator, n_proofs: int, n_epochs: int = 300,
		n_retries: int = 10):
	for _ in range(n_retries):
		reasoner = ElNaive(n_concepts, n_roles)
		for _ in tqdm(range(n_epochs)):
			for _ in range(n_retries):
				axiom = random_axiom(n_concepts, n_roles, rng)
				if axiom not in reasoner:
					break
			else:
				assert False
			reasoner.add_axiom(axiom)
			# print(reasoner.axioms)
			shortest = list(find_nontrivial_proofs(reasoner, 4))
			if len(shortest) >= n_proofs:
				return reasoner


def extract_dataset(reasoner: ElNaive, max_shortest_proof_length: int) -> Tuple[list, list, list, list]:
	true_conclusions = []
	false_conclusions = []

	for sub in list(range(reasoner.n_concepts)) + [BOT, TOP]:
		for sup in list(range(reasoner.n_concepts)) + [BOT, TOP]:
			if sup in reasoner.S[sub]:
				proof = reasoner.decode_shortest_proof(sub, sup)
				true_conclusions.append(((sub, sup), len(proof)))
			else:
				false_conclusions.append((sub, sup))

	pos_train = []
	pos_test = []

	for pair, proof_length in true_conclusions:
		if proof_length <= max_shortest_proof_length:
			pos_train.append((SUB,) + pair)
		else:
			pos_test.append((SUB,) + pair)

	positive = {x for x, _ in true_conclusions}

	neg_train = []

	for sub, sup in false_conclusions:
		if (sup, sub) not in positive:
			neg_train.append((SUB, sub, sup))

	return pos_train, neg_train, pos_test, []


def vsplit(data):
	target = [[] for _ in range(len(data[0]))]
	for row in data:
		for col, value in zip(target, row):
			col.append(value)
	return target


def split_dataset(reasoners: list[ElNaive], rng: np.random.Generator, *, complexity_threshold=2, neg_to_pos_ratio=1,
				  val_ratio=.2):
	training = []
	validation = []
	test = []

	for idx, reasoner in enumerate(reasoners):
		pos_train, neg_train, pos_test, neg_test = extract_dataset(reasoner, complexity_threshold)
		max_neg_count = neg_to_pos_ratio * len(pos_train)
		if max_neg_count < len(neg_train):
			# not using rng.choice on the list directly as it converts the data into numpy's arrays and int64s
			idxs = rng.choice(len(neg_train), size=max_neg_count, replace=False)
			neg_train = [neg_train[i] for i in idxs]
		rng.shuffle(pos_train)
		rng.shuffle(neg_train)
		n_pos_validation = int(val_ratio * len(pos_train))
		n_neg_validation = int(val_ratio * len(neg_train))
		training += [(idx, axiom, 1) for axiom in pos_train[n_pos_validation:]]
		training += [(idx, axiom, 0) for axiom in neg_train[n_neg_validation:]]
		validation += [(idx, axiom, 1) for axiom in pos_train[:n_pos_validation]]
		validation += [(idx, axiom, 0) for axiom in neg_train[:n_neg_validation]]
		test += [(idx, axiom, 1) for axiom in pos_test]
		test += [(idx, axiom, 0) for axiom in neg_test]

	print("Training", len(training), "#pos", sum(label for _, _, label in training))
	print("Validation", len(validation), "#pos", sum(label for _, _, label in validation))
	print("Test", len(test), "#pos", sum(label for _, _, label in test))

	return vsplit(training), vsplit(validation), vsplit(test)
