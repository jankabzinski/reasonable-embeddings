import itertools

from src.elpp.ElNaive import ElNaive, CR1, CR2, CR3, CR4, CR5
from src.simplefact.syntax import BOT, TOP, SUB, AND, ANY


def test_inconsistent():
	reasoner = ElNaive(2, 0)
	reasoner.add_axiom((SUB, TOP, 0))
	assert BOT not in reasoner.S[TOP]
	reasoner.add_axiom((SUB, 1, BOT))
	assert BOT not in reasoner.S[TOP]
	reasoner.add_axiom((SUB, 0, 1))
	assert BOT in reasoner.S[TOP]


def test_cr1_single_path_forward():
	reasoner = ElNaive(4, 0)
	# 0 <= 1 <= 2 <= 3
	reasoner.add_axiom((SUB, 0, 1))
	assert reasoner.S[0] == {1: {(CR1, (SUB, 0, 1), 0)}, 0: set(), TOP: set()}
	reasoner.add_axiom((SUB, 1, 2))
	assert reasoner.S[0] == {1: {(CR1, (SUB, 0, 1), 0)}, 2: {(CR1, (SUB, 1, 2), 1), (CR1, (SUB, 0, 1), 0)}, 0: set(),
							 TOP: set()}
	assert reasoner.S[1] == {2: {(CR1, (SUB, 1, 2), 1)}, 1: set(), TOP: set()}
	reasoner.add_axiom((SUB, 2, 3))
	assert reasoner.S[0] == {1: {(CR1, (SUB, 0, 1), 0)}, 2: {(CR1, (SUB, 1, 2), 1), (CR1, (SUB, 0, 1), 0)},
							 3: {(CR1, (SUB, 2, 3), 2), (CR1, (SUB, 1, 2), 1), (CR1, (SUB, 0, 1), 0)},
							 0: set(), TOP: set()}
	assert reasoner.S[1] == {2: {(CR1, (SUB, 1, 2), 1)}, 3: {(CR1, (SUB, 2, 3), 2), (CR1, (SUB, 1, 2), 1)},
							 1: set(), TOP: set()}
	assert reasoner.S[2] == {3: {(CR1, (SUB, 2, 3), 2)}, 2: set(), TOP: set()}


def test_cr1_single_path_backward():
	reasoner = ElNaive(4, 0)
	# 0 <= 1 <= 2 <= 3
	reasoner.add_axiom((SUB, 2, 3))
	assert reasoner.S[2] == {3: {(CR1, (SUB, 2, 3), 2)}, 2: set(), TOP: set()}
	reasoner.add_axiom((SUB, 1, 2))
	assert reasoner.S[2] == {3: {(CR1, (SUB, 2, 3), 2)}, 2: set(), TOP: set()}
	assert reasoner.S[1] == {2: {(CR1, (SUB, 1, 2), 1)}, 3: {(CR1, (SUB, 2, 3), 2), (CR1, (SUB, 1, 2), 1)},
							 1: set(), TOP: set()}
	reasoner.add_axiom((SUB, 0, 1))
	assert reasoner.S[0] == {1: {(CR1, (SUB, 0, 1), 0)}, 2: {(CR1, (SUB, 1, 2), 1), (CR1, (SUB, 0, 1), 0)},
							 3: {(CR1, (SUB, 2, 3), 2), (CR1, (SUB, 1, 2), 1), (CR1, (SUB, 0, 1), 0)},
							 0: set(), TOP: set()}
	assert reasoner.S[1] == {2: {(CR1, (SUB, 1, 2), 1)}, 3: {(CR1, (SUB, 2, 3), 2), (CR1, (SUB, 1, 2), 1)},
							 1: set(), TOP: set()}
	assert reasoner.S[2] == {3: {(CR1, (SUB, 2, 3), 2)}, 2: set(), TOP: set()}


def test_cr1_single_path_jumping():
	reasoner = ElNaive(4, 0)
	# 0 <= 1 <= 2 <= 3
	reasoner.add_axiom((SUB, 2, 3))
	assert reasoner.S[2] == {3: {(CR1, (SUB, 2, 3), 2)}, 2: set(), TOP: set()}
	reasoner.add_axiom((SUB, 0, 1))
	assert reasoner.S[2] == {3: {(CR1, (SUB, 2, 3), 2)}, 2: set(), TOP: set()}
	assert reasoner.S[0] == {1: {(CR1, (SUB, 0, 1), 0)}, 0: set(), TOP: set()}
	reasoner.add_axiom((SUB, 1, 2))
	assert reasoner.S[0] == {1: {(CR1, (SUB, 0, 1), 0)}, 2: {(CR1, (SUB, 1, 2), 1), (CR1, (SUB, 0, 1), 0)},
							 3: {(CR1, (SUB, 2, 3), 2), (CR1, (SUB, 1, 2), 1), (CR1, (SUB, 0, 1), 0)},
							 0: set(), TOP: set()}
	assert reasoner.S[1] == {2: {(CR1, (SUB, 1, 2), 1)}, 3: {(CR1, (SUB, 2, 3), 2), (CR1, (SUB, 1, 2), 1)},
							 1: set(), TOP: set()}
	assert reasoner.S[2] == {3: {(CR1, (SUB, 2, 3), 2)}, 2: set(), TOP: set()}


def test_cr1_two_paths():
	# 0 <= 1 <= 3
	# 0 <= 2 <= 3
	axioms = [(SUB, 0, 1), (SUB, 0, 2), (SUB, 1, 3), (SUB, 2, 3)]
	for perm in itertools.permutations(axioms):
		reasoner = ElNaive(4, 0)
		for axiom in perm:
			reasoner.add_axiom(axiom)
		assert reasoner.S[0][1] == {(CR1, (SUB, 0, 1), 0), }
		assert reasoner.S[0][2] == {(CR1, (SUB, 0, 2), 0), }
		assert reasoner.S[0][0] == set()
		assert reasoner.S[0][TOP] == set()
		assert reasoner.S[0][3] in [{(CR1, (SUB, 2, 3), 2), (CR1, (SUB, 0, 2), 0), },
									{(CR1, (SUB, 1, 3), 1), (CR1, (SUB, 0, 1), 0), }]
		assert reasoner.S[1] == {3: {(CR1, (SUB, 1, 3), 1), }, 1: set(), TOP: set()}
		assert reasoner.S[2] == {3: {(CR1, (SUB, 2, 3), 2), }, 2: set(), TOP: set()}
		assert reasoner.S[3] == {3: set(), TOP: set()}


def test_cr2():
	# 0 & 0 <= 1  -> 0 <= 1
	# 0 & 0 <= 2  -> 0 <= 2
	# 1 & 2 <= 3  -> 0 <= 3
	a001 = (SUB, (AND, 0, 0), 1)
	a002 = (SUB, (AND, 0, 0), 2)
	a123 = (SUB, (AND, 1, 2), 3)
	axioms = [a001, a002, a123]
	for perm in itertools.permutations(axioms):
		reasoner = ElNaive(4, 0)
		for axiom in perm:
			assert reasoner.S[0] != {1: {(CR2, a001, 0, 0), }, 2: {(CR2, a002, 0, 0), },
									 3: {(CR2, a123, 1, 2), (CR2, a001, 0, 0), (CR2, a002, 0, 0), }, 0: set(),
									 TOP: set()}
			reasoner.add_axiom(axiom)
		assert reasoner.S[0] == {1: {(CR2, a001, 0, 0), }, 2: {(CR2, a002, 0, 0), },
								 3: {(CR2, a123, 1, 2), (CR2, a001, 0, 0), (CR2, a002, 0, 0), }, 0: set(),
								 TOP: set()}
		assert reasoner.S[1] == {1: set(), TOP: set()}
		assert reasoner.S[2] == {2: set(), TOP: set()}
		assert reasoner.S[3] == {3: set(), TOP: set()}


def test_cr3():
	# 0 <= 0 SOME 1
	reasoner = ElNaive(2, 1)
	a = (SUB, 0, (ANY, 0, 1))
	reasoner.add_axiom(a)
	assert reasoner.R[0] == {(0, 1): {(CR3, a, 0), }}


def test_cr4():
	reasoner = ElNaive(3, 1)
	reasoner.R[0] = {(0, 1): set()}
	a = (SUB, (ANY, 0, 1), 2)
	reasoner.add_axiom(a)
	assert reasoner.S[0] == {2: {(CR4, a, 0, 0, 1, 1), }, 0: set(), TOP: set()}


def test_cr5():
	reasoner = ElNaive(2, 1)
	reasoner.R[0] = {(0, 1): set()}
	reasoner.S[1] = {BOT: set()}
	reasoner.add_axiom(None)
	assert reasoner.S[0] == {BOT: {(CR5, None, 0, 0, 1), }, 0: set(), TOP: set()}


def test_cr1_cr3_cr5():
	axioms = [(SUB, 0, (ANY, 0, 1)), (SUB, 1, (ANY, 0, 2)), (SUB, 2, (ANY, 0, 3)), (SUB, 3, BOT)]
	for perm in itertools.permutations(axioms):
		reasoner = ElNaive(4, 1)
		for axiom in perm:
			reasoner.add_axiom(axiom)
		assert BOT in reasoner.S[0]


def test_decode_shortest_cr1_cr3_cr5():
	axioms = [(SUB, 0, (ANY, 0, 1)), (SUB, 1, (ANY, 0, 2)), (SUB, 2, (ANY, 0, 3)), (SUB, 3, BOT)]
	reasoner = ElNaive(4, 1)
	for axiom in axioms:
		reasoner.add_axiom(axiom)
	proof = reasoner.decode_shortest_proof(0, BOT)
	assert set(proof) == {
		(CR5, None),
		(CR3, axioms[0]),
		(CR5, None),
		(CR3, axioms[1]),
		(CR5, None),
		(CR3, axioms[2]),
		(CR1, axioms[3])
		, }


def test_decode_cycle():
	reasoner = ElNaive(4, 0)
	reasoner.add_axiom((SUB, 0, 1))
	reasoner.add_axiom((SUB, 1, 2))
	reasoner.add_axiom((SUB, 2, 3))
	reasoner.add_axiom((SUB, 3, 0))
	proof = reasoner.decode_shortest_proof(0, 3)
	print(proof)
