from src.simplefact.syntax import BOT, TOP, SUB, AND, ANY


def to_turtle(axioms):
	classes = {}
	roles = {}

	def c(n):
		if n == BOT:
			return "owl:Nothing"
		if n == TOP:
			return "owl:Thing"
		return classes.setdefault(n, f"<http://example.com/C{n}>")

	def r(n):
		return roles.setdefault(n, f"<http://example.com/R{n}>")

	def translate(expr):
		if isinstance(expr, int):
			return c(expr)
		op, lhs, rhs = expr
		if op == SUB:
			return f"{translate(lhs)} rdfs:subClassOf {translate(rhs)} .\n"
		if op == AND:
			return f"[ a owl:Class; owl:intersectionOf ({translate(lhs)} {translate(rhs)}) ]"
		if op == ANY:
			return f"[ a owl:Restriction; owl:onProperty {r(lhs)}; owl:someValuesFrom {translate(rhs)} ]"
		assert False, expr

	prefix = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
	"""
	text = ""
	for axiom in axioms:
		text += translate(axiom)
	for iri in classes.values():
		prefix += f"{iri} a owl:Class .\n"
	for iri in roles.values():
		prefix += f"{iri} a owl:ObjectProperty .\n"
	return prefix + text
