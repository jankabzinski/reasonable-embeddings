import heapq
import itertools
from typing import Tuple

import numpy as np

from src.simplefact.syntax import BOT, TOP, SUB, AND, ANY

CR1 = 'cr1'
CR2 = 'cr2'
CR3 = 'cr3'
CR4 = 'cr4'
CR5 = 'cr5'


class MyPriorityQueue:
	def __init__(self):
		self._heap = []

	def put(self, item):
		heapq.heappush(self._heap, item)

	def get(self):
		return heapq.heappop(self._heap)

	def empty(self):
		return len(self._heap) == 0


class ElNaive:
	S: dict[int, dict[int, set]]
	R: list[dict[tuple[int, int], set]]

	def __init__(self, n_concepts: int, n_roles: int):
		self.n_concepts = n_concepts
		self.n_roles = n_roles
		self.S = {c: {c: set(), TOP: set()} for c in range(min(TOP, BOT), self.n_concepts)}
		self.R = [{} for r in range(self.n_roles)]
		self.axioms = {CR1: set(), CR2: set(), CR3: set(), CR4: set()}

	def __contains__(self, item):
		return any(item in v for v in self.axioms.values())

	def cr1(self, axiom: Tuple[int, int, int]) -> bool:
		op, cp, d = axiom
		assert op == SUB
		assert isinstance(cp, int)
		assert isinstance(d, int)
		modified = False
		for c, sc in self.S.items():
			if cp in sc:
				cp_proof = sc[cp]
				proof = {(CR1, axiom, cp)} | cp_proof
				if d not in sc or len(proof) < len(sc[d]):
					sc[d] = proof
					modified = True
		return modified

	def cr2(self, axiom: Tuple[int, Tuple[int, int, int], int]) -> bool:
		_, (_, c1, c2), d = axiom
		modified = False
		for c, sc in self.S.items():
			if c1 in sc and c2 in sc:
				c1_proof = sc[c1]
				c2_proof = sc[c2]
				proof = {(CR2, axiom, c1, c2)} | c1_proof | c2_proof
				if d not in sc or len(proof) < len(sc[d]):
					sc[d] = proof
					modified = True
		return modified

	def cr3(self, axiom: Tuple[int, int, Tuple[int, int, int]]) -> bool:
		_, cp, (_, r, d) = axiom
		rr = self.R[r]
		modified = False
		for c, sc in self.S.items():
			if cp in sc:
				cp_proof = sc[cp]
				proof = {(CR3, axiom, cp)} | cp_proof
				if (c, d) not in rr or len(proof) < len(rr[(c, d)]):
					rr[(c, d)] = proof
					modified = True
		return modified

	def cr4(self, axiom: Tuple[int, Tuple[int, int, int], int]) -> bool:
		_, (_, r, dp), e = axiom
		modified = False
		for (c, d), cd_proof in self.R[r].items():
			if dp in self.S[d]:
				sc = self.S[c]
				dp_proof = self.S[d][dp]
				proof = {(CR4, axiom, r, c, d, dp)} | cd_proof | dp_proof
				if e not in sc or len(proof) < len(sc[e]):
					sc[e] = proof
					modified = True
		return modified

	def cr5(self) -> bool:
		modified = False
		for r in range(self.n_roles):
			for (c, d), cd_proof in self.R[r].items():
				if BOT in self.S[d]:
					sc = self.S[c]
					bot_proof = self.S[d][BOT]
					proof = {(CR5, None, r, c, d)} | cd_proof | bot_proof
					if BOT not in sc or len(proof) < len(sc[BOT]):
						sc[BOT] = proof
						modified = True
		return modified

	def decode_shortest_proof(self, c, d, r=None):
		if r is None:
			return tuple(x[:2] for x in self.S[c][d])
		else:
			return tuple(x[:2] for x in self.R[r][(c, d)])

	def add_axiom(self, axiom):
		# TODO CR10 and CR11
		if axiom is not None:
			left_atomic = isinstance(axiom[1], int)
			right_atomic = isinstance(axiom[2], int)
			assert axiom[0] == SUB
			if right_atomic:
				if left_atomic:
					self.axioms[CR1].add(axiom)
				else:
					op, subleft, subright = axiom[1]
					if op == AND:
						self.axioms[CR2].add(axiom)
					elif op == ANY:
						self.axioms[CR4].add(axiom)
					else:
						raise NotImplemented()
			else:
				op, supleft, supright = axiom[2]
				if left_atomic and op == ANY:
					self.axioms[CR3].add(axiom)
				else:
					raise NotImplemented()
		modified = True
		while modified:
			modified = False
			for axiom in self.axioms[CR1]:
				modified |= self.cr1(axiom)
			for axiom in self.axioms[CR2]:
				modified |= self.cr2(axiom)
			for axiom in self.axioms[CR3]:
				modified |= self.cr3(axiom)
			for axiom in self.axioms[CR4]:
				modified |= self.cr4(axiom)
			modified |= self.cr5()


def find_nontrivial_proofs(reasoner: ElNaive, min_shortest_proof_length: int):
	for c in range(reasoner.n_concepts):
		if BOT in reasoner.S[c].keys():
			continue
		for d in reasoner.S[c].keys():
			shortest_proof = reasoner.decode_shortest_proof(c, d)
			if len(shortest_proof) >= min_shortest_proof_length:
				yield (c, d), shortest_proof

