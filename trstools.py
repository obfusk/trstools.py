#!/usr/bin/python

# --                                                            ; {{{1
#
# File        : trstools.py
# Maintainer  : Felix C. Stegerman <flx@obfusk.net>
# Date        : 2016-05-18
#
# Copyright   : Copyright (C) 2016  Felix C. Stegerman
# Version     : v0.1.1
# License     : GPLv3+
#
# --                                                            ; }}}1

                                                                # {{{1
r"""
Python (2+3) term rewriting tools

Examples
========

>>> import trstools as T

>>> r1  = T.rule("f(x,h(x)) -> f(x,x)")
>>> r2  = T.rule("f(g(x),y) -> f(x,h(y))")
>>> r3  = T.rule("g(h(x))   -> h(g(x))")

>>> rs  = [r1, r2, r3]

>>> t1  = T.term("f(g(h(x)),y)")
>>> t2  = T.term("f(h(h(x)),y)")
>>> t3  = T.term("f(g(h(g(h(x)))),y)")

>>> for u in t1.subterms(): print(u)
f(g(h(x)),y)
g(h(x))
h(x)
x
y

>>> tuple(sorted(T.variables(t1)))
(x, y)

>>> sigma = dict(x = "z", y = "g(x)")
>>> t1.substitute(sigma)
f(g(h(z)),g(x))

>>> for a in t3.applyrec_(rs, ignore_seen = False, bfs = False):
...   print("%-17s" % " -> ".join(map(str,a.applied_rules)), a.term)
1                 f(h(g(h(x))),h(y))
1 -> 2            f(h(h(g(x))),h(y))
2                 f(h(g(g(h(x)))),y)
2 -> 2            f(h(g(h(g(x)))),y)
2 -> 2 -> 2       f(h(h(g(g(x)))),y)
2                 f(g(h(h(g(x)))),y)
2 -> 1            f(h(h(g(x))),h(y))
2 -> 2            f(h(g(h(g(x)))),y)
2 -> 2 -> 2       f(h(h(g(g(x)))),y)

>>> for u in t3.normalforms(rs): print(u)
f(h(h(g(x))),h(y))
f(h(h(g(g(x)))),y)

>>> for u, v in T.critical_pairs(rs, trivial = False):
...   print("[ %-16s, %-12s ]" % (u, v))
[ f(x,h(h(g(x)))) , f(g(x),g(x)) ]
[ f(h(g(x)),z)    , f(h(x),h(z)) ]

>>> for p in T.critical_pairs(rs):
...   print(is_convergent_pair(p, rs), is_trivial_pair(p))
True True
False False
True True
False False
True True

>>> T.show_tree(t3, rs)
f(g(h(g(h(x)))),y)
  --1-->  f(h(g(h(x))),h(y))
    --2-->  f(h(h(g(x))),h(y))  NF
  --2-->  f(h(g(g(h(x)))),y)
    --2-->  f(h(g(h(g(x)))),y)
      --2-->  f(h(h(g(g(x)))),y)  NF
  --2-->  f(g(h(h(g(x)))),y)
    --1-->  f(h(h(g(x))),h(y))  NF
    --2-->  f(h(g(h(g(x)))),y)
      --2-->  f(h(h(g(g(x)))),y)  NF
"""
# ... TODO ...
                                                                # }}}1

from __future__ import print_function

import argparse, collections, os, subprocess, sys, tempfile
import pyparsing as P

if sys.version_info.major == 2:                                 # {{{1
  def iteritems(x): return x.iteritems()
else:
  def iteritems(x): return x.items()
                                                                # }}}1

__version__       = "0.1.1"

def main(*args):                                                # {{{1
  p = argument_parser(); n = p.parse_args(args)
  if n.test:
    import doctest
    doctest.testmod(verbose = n.verbose)
    return 0
  incompat_args = [n.normalforms, n.critical_pairs, n.tree, n.graph]
  if  len([ x for x in incompat_args if x ]) > 1        or \
      (n.output and not n.graph)                        or \
      (n.trivial is not None and not n.critical_pairs)  or \
      (n.mark_nf and not (n.tree or n.graph)):
    print("{}: error: incompatible arguments".format(p.prog),
          file = sys.stderr)                                    # TODO
    return 2
  rules = [ rule(r) for r in n.rules ]
  if n.rules_from: rules += list(rules_from_file(n.rules_from))
  if n.normalforms:
    for u in term(n.normalforms).normalforms(rules): print(u)
  elif n.critical_pairs:
    for u, v in critical_pairs(rules, trivial = n.trivial):
      print("[ %s, %s ]" % (u, v))
  elif n.tree:
    show_tree(term(n.tree), rules, show_nf = n.mark_nf)
  elif n.graph:
    show_or_save_digraph(term(n.graph), rules, fname = n.output,
                         show_nf = n.mark_nf)
  else:
    print("{}: error: missing command".format(p.prog),
          file = sys.stderr)
    return 1
  return 0
                                                                # }}}1

def argument_parser():                                          # {{{1
  p = argparse.ArgumentParser(description = "trstools")
  p.add_argument("--rule", dest = "rules", metavar = "RULE",
                 action = "append",
                 help = "specify a TRS rule; "
                        "can be used multiple times")
  p.add_argument("--rules-from", metavar = "FILE",
                 help = "specify TRS rules in a file")
  p.add_argument("--normalforms", metavar = "TERM",
                 help = "show normal forms of TERM")
  p.add_argument("--critical-pairs", action = "store_true",
                 help = "show critical pairs of TRS")
  p.add_argument("--trivial", action = "store_true",
                 help = "show trivial critical pairs")
  p.add_argument("--tree", metavar = "TERM",
                 help = "show tree of reductions of TERM")
  p.add_argument("--mark-nf", action = "store_true",
                 help = "mark normal forms in tree or graph")
  p.add_argument("--graph", metavar = "TERM",
                 help = "show (or save) graph of reductions of TERM")
  p.add_argument("--output", metavar = "FILE",
                 help = "save graph to FILE instead of showing it")
  p.add_argument("--version", action = "version",
                 version = "%(prog)s {}".format(__version__))
  p.add_argument("--test", action = "store_true",
                 help = "run tests (not trstools)")
  p.add_argument("--verbose", "-v", action = "store_true",
                 help = "run tests verbosely")
  p.set_defaults(rules = [], trivial = None)
  return p
                                                                # }}}1

def rules_from_file(fname):
  with open(fname) as f:
    for line in ( line.strip() for line in f ):
      if line and not line.startswith("#"): yield rule(line)

class Immutable(object):                                        # {{{1

  """immutable base class"""

  __slots__ = []

  args_are_mandatory = False

  @property
  def ___slots(self):
    return [x for x in self.__slots__ if not x.startswith("_")]

  def __init__(self, data = None, **kw):
    x = data if data is not None else {}; x.update(kw)
    ks = set(x.keys()); ss = set(self.___slots)
    for k in self.___slots:
      if k in x:
        self._Immutable___set(k, x[k]); del x[k]
      else:
        self._Immutable___set(k, None)
    if len(x):
      raise TypeError("unknown keys: {}".format(", ".join(x.keys())))
    if self.args_are_mandatory and ks != ss:
      raise TypeError("missing keys: {}".format(", ".join(ss - ks)))

  def ___set(self, k, v):
    super(Immutable, self).__setattr__(k, v)

  def __setattr__(self, k, v):
    if k in self.___slots:
      raise AttributeError(
        "'{}' object attribute '{}' is read-only".format(
          self.__class__.__name__, k
        )
      )
    else:
      raise AttributeError(
        "'{}' object has no attribute '{}'".format(
          self.__class__.__name__, k
        )
      )

  def copy(self, **kw):
    return type(self)(dict(self.iteritems()), **kw)

  def iteritems(self):
    return ((k, getattr(self, k)) for k in self.___slots)

  if sys.version_info.major == 2:
    def items(self):
      return list(self.iteritems())
  else:
    def items(self):
      return self.iteritems()

  def __eq__(self, rhs):
    if not isinstance(rhs, type(self)): return NotImplemented
    return dict(self.iteritems()) == dict(rhs.iteritems())

  def __lt__(self, rhs):
    if not isinstance(rhs, type(self)): return NotImplemented
    return sorted(self.iteritems()) < sorted(rhs.iteritems())

  def __le__(self, rhs):
    if not isinstance(rhs, type(self)): return NotImplemented
    return sorted(self.iteritems()) <= sorted(rhs.iteritems())

  def __gt__(self, rhs):
    if not isinstance(rhs, type(self)): return NotImplemented
    return sorted(self.iteritems()) > sorted(rhs.iteritems())

  def __ge__(self, rhs):
    if not isinstance(rhs, type(self)): return NotImplemented
    return sorted(self.iteritems()) >= sorted(rhs.iteritems())

  def __repr__(self):
    return '{}({})'.format(
      self.__class__.__name__,
      ", ".join("{} = {}".format(k, repr(v))
                for (k,v) in self.iteritems())
    )

  def __hash__(self):
    return hash(tuple(self.iteritems()))
                                                                # }}}1

FUNCTIONS   = P.Word("fgh", P.alphas + "'")
VARIABLES   = P.Word("xyz", P.alphas + "'")
PRE1, PRE2  = "xy"
EASYFUNCS   = "x y z x' y' z'".split()

class Function(Immutable):                                      # {{{1
  """Function with zero-or-more arguments."""

  __slots__ = "name args".split()

  def __init__(self, name, *args):
    super(Function, self).__init__(name = name, args = args)

  def copy(self, name = None, args = None):
    if name is None: name = self.name
    if args is None: args = self.args
    return type(self)(name, *args)

  # TODO
  def with_arg(self, i, x):
    args = list(self.args); args[i] = x
    return self.copy(args = args)

  def __repr__(self):
    return self.name + "(" + ",".join(map(repr, self.args)) + ")"
                                                                # }}}1

class Variable(Immutable):                                      # {{{1
  """Variable."""

  __slots__ = "name".split()

  def __init__(self, name):
    super(Variable, self).__init__(name = name)

  def copy(self, name = None):
    if name is None: name = self.name
    return type(self)(name)

  def __repr__(self):
    return self.name
                                                                # }}}1

def isfunc(x): return isinstance(x, Function)
def isvar (x): return isinstance(x, Variable)
def isterm(x): return isfunc(x) or isvar(x)

class Rule(Immutable):                                          # {{{1
  """Rule (left -> right)."""

  __slots__ = "left right".split()

  def __init__(self, left, right):
    super(Rule, self).__init__(left = left, right = right)

  def __repr__(self):
    return repr(self.left) + " -> " + repr(self.right)
                                                                # }}}1

def term(x):                                                    # {{{1
  r"""
  Turn string or parse result into a nested Function/Variable tree.

  >>> import trstools as T
  >>> t1  = T.term("f(g(h(x)),y)")
  >>> f,v = T.Function, T.Variable
  >>> t2  = f("f",f("g",f("h",v("x"))),v("y"))
  >>> t1 == t2
  True
  """

  if isterm(x): return x
  if not isinstance(x, P.ParseResults): x = parse_term(x)
  if "varname" in x: return Variable(x.varname)
  return Function(x.funcname, *map(term, x.subterms))
                                                                # }}}1

def parse_term(t, func = FUNCTIONS, var = VARIABLES):           # {{{1
  """Parse a term."""

  lp, rp  = P.Literal("("), P.Literal(")")
  fu, va  = func("funcname"), var("varname")
  expr    = P.Forward()
  st      = P.delimitedList(P.Group(expr), ",")
  expr << ( va | fu + lp + P.Optional(st("subterms")) + rp )
  return expr.parseString(t, True)
                                                                # }}}1

def rule(l, r = None):
  """Create/parse a rule."""
  if isinstance(l, Rule): return l
  if r is None: l, r = l.split("->")
  return Rule(term(l), term(r))

Subterm = collections.namedtuple("Subterm", "term wrap")

def subterms_(t, proper = False, variables = True):             # {{{1
  r"""
  Iterate over subterms (and functions that can wrap a substitute term
  back into the term in its place) of a term.

  >>> import trstools as T
  >>> for t, w in T.term("f(g(h(x)),y)").subterms_():
  ...   print(t, w(T.term("z")))
  f(g(h(x)),y) z
  g(h(x)) f(z,y)
  h(x) f(g(z),y)
  x f(g(h(z)),y)
  y f(g(h(x)),z)
  """

  def g(i, f):
    def h(x): return t.with_arg(i, f(x))
    return h
  if isfunc(t):
    if not proper:
      yield Subterm(t, id_)
    for i, u in enumerate(t.args):
      for v, f in subterms_(u, variables = variables):
        yield Subterm(v, g(i, f))
  elif variables and not proper:
    yield Subterm(t, id_)
                                                                # }}}1

def subterms(t, proper = False, variables = True):
  """Iterate over subterms of a term."""
  for u in subterms_(t, proper, variables): yield u.term

def variables(t):
  """The set of variales of a term."""
  return set( u for u in subterms(t) if isvar(u) )

def substitute(t, sigma):                                       # {{{1
  r"""
  Substitute terms for variables.

  >>> import trstools as T
  >>> sigma = dict(x = "z", y = "g(x)")
  >>> T.term("f(g(h(x)),y)").substitute(sigma)
  f(g(h(z)),g(x))
  """

  if isvar(t): return term(sigma.get(t.name, t))
  return t.copy(args = [ substitute(u, sigma) for u in t.args ])
                                                                # }}}1

# TODO
def applyrule(t, r, _vars = None):                              # {{{1
  r"""
  Apply a rule to a term (if possible); returns None otherwise.

  >>> import trstools as T
  >>> r = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> T.term("f(g(h(x)),y)").applyrule(r)
  f(h(x),h(y))
  """

  lhs, rhs = r.left, r.right
  if _vars is None: _vars = {}
  if isvar(lhs):
    if lhs.name not in _vars or _vars[lhs.name] == t:
      _vars[lhs.name] = t
      return t
  elif isfunc(t):
    if t.name == lhs.name:
      if len(t.args) != len(lhs.args):
        raise ValueError("functions differ in arity!")          # TODO
      for u, v in zip(t.args, lhs.args):
        if not applyrule(u, Rule(v, None), _vars): return None
      return substitute(rhs, _vars) if rhs is not None else True
  return None
                                                                # }}}1
Application = collections.namedtuple("Application", "term left "
                                     "subleft rule applied_rules")

def apply1_(t, rules, already_applied = ()):                    # {{{1
  r"""
  Iterate over applications of rules to (subterms of) a term.

  >>> import trstools as T
  >>> r1 = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> r2 = T.rule("g(h(x))   -> h(g(x))")
  >>> for a in T.term("f(g(h(x)),y)").apply1_([r1,r2]):
  ...   print("%-17s" % " -> ".join(map(str,a.applied_rules)), a.term)
  0                 f(h(x),h(y))
  1                 f(h(g(x)),y)
  """

  for u, f in subterms_(t):
    for i, r in enumerate(rules):
      v = applyrule(u, r)
      if v: yield Application(f(v), t, u, r, already_applied + (i,))
                                                                # }}}1

def apply1(t, rules):
  """Iterate over terms resulting from applications of rules to
  (subterms of) a term."""
  for u in apply1_(t, rules): yield u.term

def applyrec_(t, rules, n = None, ignore_seen = True,
              bfs = True):                                      # {{{1
  """
  Iterate over recursive applications of rules to (subterms of) a
  term.

  >>> import trstools as T
  >>> r1  = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> r2  = T.rule("g(h(x))   -> h(g(x))")
  >>> t   = T.term("f(g(h(g(h(x)))),y)")

  >>> for a in t.applyrec_([r1,r2]):
  ...   print("%-17s" % " -> ".join(map(str,a.applied_rules)), a.term)
  0                 f(h(g(h(x))),h(y))
  1                 f(h(g(g(h(x)))),y)
  1                 f(g(h(h(g(x)))),y)
  0 -> 1            f(h(h(g(x))),h(y))
  1 -> 1            f(h(g(h(g(x)))),y)
  1 -> 1 -> 1       f(h(h(g(g(x)))),y)

  >>> for a in t.applyrec_([r1,r2], ignore_seen = False, bfs = False):
  ...   print("%-17s" % " -> ".join(map(str,a.applied_rules)), a.term)
  0                 f(h(g(h(x))),h(y))
  0 -> 1            f(h(h(g(x))),h(y))
  1                 f(h(g(g(h(x)))),y)
  1 -> 1            f(h(g(h(g(x)))),y)
  1 -> 1 -> 1       f(h(h(g(g(x)))),y)
  1                 f(g(h(h(g(x)))),y)
  1 -> 0            f(h(h(g(x))),h(y))
  1 -> 1            f(h(g(h(g(x)))),y)
  1 -> 1 -> 1       f(h(h(g(g(x)))),y)
  """

  terms, seen = collections.deque([(0, t, ())]), set([t])
  while terms:
    i, u, a = terms.popleft()
    if i != 0:
      yield u; u = u.term
    append = []
    for v in apply1_(u, rules, a):
      if  (not ignore_seen or v.term not in seen) and \
          (n is None or i < n):
        append += [(i+1,v,v.applied_rules)]
        if ignore_seen: seen.add(v.term)
    if bfs: terms.extend(append)
    else:   terms.extendleft(reversed(append))
                                                                # }}}1

def applyrec(*a, **kw):
  """Iterate over terms resulting from recursive applications of rules
  to (subterms of) a term."""
  for u in applyrec_(*a, **kw): yield u.term

def isnormalform(t, rules):
  """Is term t a normal form?"""
  return len(list(apply1(t, rules))) == 0

def normalforms(t, rules):                                      # {{{1
  """
  Iterate over normal forms of term (for terminating TRS).

  >>> import trstools as T
  >>> r1  = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> r2  = T.rule("g(h(x))   -> h(g(x))")
  >>> t   = T.term("f(g(h(g(h(x)))),y)")
  >>> for u in t.normalforms([r1,r2]): print(u)
  f(h(h(g(x))),h(y))
  f(h(h(g(g(x)))),y)
  """

  if isnormalform(t, rules): yield t
  else:
    for u in applyrec(t, rules):
      if isnormalform(u, rules): yield u
                                                                # }}}1

# TODO
def distinct_variable_funcs(u, v, pre1 = PRE1, pre2 = PRE2,
                            easy = EASYFUNCS):                  # {{{1
  """Create functions to make the variables in u and v distinct by
  substitution."""

  v1, v2    = variables(u), variables(v)
  if not (v1 & v2): return id_, id_
  if len(v1) + len(v2) <= len(easy):
    s1      = dict(zip(sorted( x.name for x in v1 ), easy[:len(v1)]))
    s2      = dict(zip(sorted( x.name for x in v2 ), easy[len(v1):]))
  else:
    f       = lambda var, p: dict( (x.name,p+x.name) for x in var )
    s1, s2  = f(v1, pre1), f(v2, pre2)
  f1  , f2  = lambda x: substitute(x, s1), \
              lambda x: substitute(x, s2)
  return f1, f2
                                                                # }}}1

Unification = collections.namedtuple("Unification",
                                     "u v sigma u_sigma v_sigma")

# TODO
def _naive_unify(u, v, sigma):                                  # {{{1
  if isvar(u):
    if u.name not in sigma or sigma[u.name] == v:
      sigma[u.name] = v
      return sigma
  elif isvar(v):
    if v.name not in sigma or sigma[v.name] == u:
      sigma[v.name] = u
      return sigma
  else:
    if u.name == v.name:
      if len(u.args) != len(v.args):
        raise ValueError("functions differ in arity!")          # TODO
      for w1, w2 in zip(u.args, v.args):
        if not _naive_unify(w1, w2, sigma): return None
      return sigma
  return None
                                                                # }}}1

# TODO
def _fix_sigma(sigma, n):                                       # {{{1
  s1, s2 = sigma, None
  while s1 != s2:
    if n <= 0: raise RuntimeError("recurses!")                  # TODO
    n  -= 1
    s2  = s1
    s1  = dict( (k, substitute(v, s1)) for k, v in iteritems(s1) )
  return s1
                                                                # }}}1

def unify(u, v, already_distinct = False):                      # {{{1
  """
  Find a sigma that unifies two terms.

  >>> import trstools as T
  >>> r1 = T.rule("f(x,h(x)) -> f(x,x)")
  >>> r2 = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> r3 = T.rule("g(h(x))   -> h(g(x))")

  >>> u1 = T.unify(r1.left.substitute(dict(x = "z")), r1.left)
  >>> print(u1.u_sigma); print(u1.v_sigma)
  f(x,h(x))
  f(x,h(x))

  >>> u2 = T.unify(r1.left, r2.left)
  >>> print(u2.u_sigma); print(u2.v_sigma)
  f(g(y),h(g(y)))
  f(g(y),h(g(y)))

  >>> not T.unify(r1.left, r3.left)
  True
  """

  if not already_distinct:
    f, g  = distinct_variable_funcs(u, v)
    u, v  = f(u), g(v)
  sigma = _naive_unify(u, v, {})
  if sigma:
    sigma = _fix_sigma(sigma, len(sigma))
    return Unification(u, v, sigma, substitute(u, sigma),
                                    substitute(v, sigma))
  return None
                                                                # }}}1

def critical_pairs(rules, trivial = True):                      # {{{1
  """
  Iterate over critical pairs of a (terminating) TRS.

  >>> import trstools as T
  >>> r1 = T.rule("f(x,h(x)) -> f(x,x)")
  >>> r2 = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> r3 = T.rule("g(h(x))   -> h(g(x))")

  >>> for u, v in T.critical_pairs([r1, r2, r3]):
  ...   print("[ %-16s, %-12s ]" % (u, v))
  [ f(y,y)          , f(y,y)       ]
  [ f(x,h(h(g(x)))) , f(g(x),g(x)) ]
  [ f(z,h(x'))      , f(z,h(x'))   ]
  [ f(h(g(x)),z)    , f(h(x),h(z)) ]
  [ h(g(y))         , h(g(y))      ]

  >>> for u, v in T.critical_pairs([r1, r2, r3], trivial = False):
  ...   print("[ %-16s, %-12s ]" % (u, v))
  [ f(x,h(h(g(x)))) , f(g(x),g(x)) ]
  [ f(h(g(x)),z)    , f(h(x),h(z)) ]
  """

  for i, r1 in enumerate(rules):
    for j, r2 in enumerate(rules):
      f, g  = distinct_variable_funcs(r1.left, r2.left)
      u, v  = f(r1.left), g(r2.left)
      p     = i < j if trivial else i <= j
      for t, h in subterms_(v, proper = p, variables = False):
        uni = unify(u, t, already_distinct = True)
        if uni:
          yield substitute(h(f(r1.right)), uni.sigma), \
                substitute(  g(r2.right) , uni.sigma)
                                                                # }}}1

def is_convergent_pair(p, rules):
  """Is the critical pair convergent?"""
  u , v   = p
  n1, n2  = list(normalforms(u, rules)), list(normalforms(v, rules))
  return n1 == n2

def is_trivial_pair(p):
  """Is the critical pair trivial?"""
  u, v = p; return u == v

# TODO
def show_tree_(t, applications, rules, show_nf = True):
  print(t)
  for a in applications:
    s, r, t = "  "*len(a.applied_rules), a.applied_rules[-1], a.term
    n       = "  NF" if show_nf and isnormalform(t, rules) else ""
    print("{}-{:-^3}->  {}{}".format(s, r, t, n))

def show_tree(t, rules, show_nf = True, ignore_seen = False,
              bfs = False):
  """Show tree of applications."""
  show_tree_(t, t.applyrec_(rules, ignore_seen = ignore_seen,
                            bfs = bfs), rules, show_nf)

def digraph(t, applications, rules, show_nf = True):            # {{{1
  """Iterate over lines of digraph of applications."""
  yield "digraph G {"
  yield "  node [shape=plaintext];"
  for a in applications:
    n = " NF" if show_nf and isnormalform(a.term, rules) else ""
    yield '  "%s" -> "%s%s" [label="%d"];' % \
      (a.left, a.term, n, a.applied_rules[-1])
  yield "}"
                                                                # }}}1

DOTCMD  = (os.environ.get("DOT_COMMAND","") or "dot -Tpng").split()
OPENCMD = (os.environ.get("IMAGE_VIEWER","") or "xdg-open").split()

def show_or_save_digraph_(g, fname = None):                     # {{{1
  """Write digraph to tempfile, convert to png and save or open."""
  with tempfile.NamedTemporaryFile() as f1:
    for line in g: f1.write(line.encode("utf-8"))
    f1.flush()
    with (open(fname, "w") if fname else
          tempfile.NamedTemporaryFile()) as f2:
      subprocess.check_call(DOTCMD + [f1.name], stdout = f2)
      if not fname:
        f2.flush(); subprocess.check_call(OPENCMD + [f2.name])
        print("Press return..."); sys.stdin.readline()
                                                                # }}}1

def show_or_save_digraph(t, rules, fname = None, show_nf = True,
                         ignore_seen = False, bfs = True):
  apps = t.applyrec_(rules, bfs = bfs, ignore_seen = ignore_seen)
  show_or_save_digraph_(digraph(t, apps, rules, show_nf), fname)

def show_digraph(t, rules, **kw):
  """Create digraph of applications using dot and show it using
  xdg-open."""
  show_or_save_digraph(t, rules, **kw)

def save_digraph(fname, t, rules, **kw):
  """Create digraph of applications using dot and save it."""
  show_or_save_digraph(t, rules, fname, **kw)

for f in "subterms_ subterms variables substitute       \
          applyrule apply1_ apply1 applyrec_ applyrec   \
          isnormalform normalforms".split():
  setattr(Variable, f, vars()[f])
  setattr(Function, f, vars()[f])

id_ = lambda x: x

if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:]))

# vim: set tw=70 sw=2 sts=2 et fdm=marker :
