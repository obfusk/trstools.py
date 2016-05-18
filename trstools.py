#!/usr/bin/python

# --                                                            ; {{{1
#
# File        : trstools.py
# Maintainer  : Felix C. Stegerman <flx@obfusk.net>
# Date        : 2016-05-18
#
# Copyright   : Copyright (C) 2016  Felix C. Stegerman
# Version     : v0.0.1
# License     : GPLv3+
#
# --                                                            ; }}}1

                                                                # {{{1
r"""
Python (2+3) term rewriting tools

Examples
========

>>> import trstools as T

... TODO ...

"""
                                                                # }}}1

from __future__ import print_function

import argparse, collections, pyparsing as P, sys

if sys.version_info.major == 2:                                 # {{{1
  pass
else:
  xrange = range
                                                                # }}}1

__version__       = "0.0.1"

# TODO
def main(*args):                                                # {{{1
  p = argument_parser(); n = p.parse_args(args)
  if n.test:
    import doctest
    doctest.testmod(verbose = n.verbose)
    return 0
  # ... TODO ...
  return 0
                                                                # }}}1

# TODO
def argument_parser():                                          # {{{1
  p = argparse.ArgumentParser(description = "trstools")
  p.add_argument("--version", action = "version",
                 version = "%(prog)s {}".format(__version__))
  p.add_argument("--test", action = "store_true",
                 help = "run tests (not trstools)")
  p.add_argument("--verbose", "-v", action = "store_true",
                 help = "run tests verbosely")
  # ... TODO ...
  return p
                                                                # }}}1

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

class Function(Immutable):                                      # {{{1
  """Function with zero-or-more arguments."""

  __slots__ = "name args".split()

  def __init__(self, name, *args):
    super(Function, self).__init__(name = name, args = args)

  def copy(self, args = None):
    if args is None: args = self.args
    return type(self)(self.name, *args)

  # TODO
  def with_arg(self, i, x):
    args = list(self.args); args[i] = x
    return self.copy(args)

  def __repr__(self):
    return self.name + "(" + ",".join(map(repr, self.args)) + ")"
                                                                # }}}1

class Variable(Immutable):                                      # {{{1
  """Variable."""

  __slots__ = "name".split()

  def __init__(self, name):
    super(Variable, self).__init__(name = name)

  def copy(self):
    return super(Variable, self).copy()

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

class Ruleset(Immutable):                                       # {{{1
  """Set of rules."""

  __slots__ = "rules".split()

  def __init__(*rules):
    super(Ruleset, self).__init__(rules = tuple(map(rule, rules)))
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
  elif isinstance(x, P.ParseResults):
    if "varname" in x: return Variable(x.varname)
    return Function(x.funcname, *map(term, x.subterms))
  else: return term(parse_term(x))
                                                                # }}}1

FUNCTIONS = P.Word("fgh", P.alphas + "'")
VARIABLES = P.Word("xyz", P.alphas + "'")

def parse_term(t, func = FUNCTIONS, var = VARIABLES):           # {{{1
  """Parse a term."""

  lp, rp  = P.Literal("("), P.Literal(")")
  fu, va  = func("funcname"), var("varname")
  expr    = P.Forward()
  st      = P.delimitedList(P.Group(expr), ",")
  expr << ( va | fu + lp + P.Optional(st("subterms")) + rp )
  return expr.parseString(t)
                                                                # }}}1

def rule(l, r = None):
  """Create/parse a rule."""
  if isinstance(l, Rule): return l
  if r is None: l, r = l.split("->")
  return Rule(term(l), term(r))

def ruleset():
  """..."""

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
      yield Subterm(t, lambda x: x)
    for i, u in enumerate(t.args):
      for v, f in subterms_(u, variables = variables):
        yield Subterm(v, g(i, f))
  elif variables and not proper:
    yield Subterm(t, lambda x: x)
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
  return t.copy([ substitute(u, sigma) for u in t.args ])
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
        raise("functions differ in arity!")                     # TODO
      for i, u in enumerate(t.args):
        r_ = Rule(lhs.args[i], None)
        if not applyrule(u, r_, _vars): return None
      return substitute(rhs, _vars) if rhs is not None else True
  return None
                                                                # }}}1

Application = collections.namedtuple("Application",
                                     "term left rule applied_rules")

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
      if v: yield Application(f(v), u, r, already_applied + (i,))
                                                                # }}}1

def apply1(t, rules):
  """Iterate over terms resulting from applications of rules to
  (subterms of) a term."""
  for u in apply1_(t, rules): yield u.term

def apply_(t, rules, n = None, ignore_seen = True, bfs = True): # {{{1
  """
  Iterate over recursive applications of rules to (subterms of) a
  term.

  >>> import trstools as T
  >>> r1  = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> r2  = T.rule("g(h(x))   -> h(g(x))")
  >>> t   = T.term("f(g(h(g(h(x)))),y)")

  >>> for a in t.apply_([r1,r2]):
  ...   print("%-17s" % " -> ".join(map(str,a.applied_rules)), a.term)
  0                 f(h(g(h(x))),h(y))
  1                 f(h(g(g(h(x)))),y)
  1                 f(g(h(h(g(x)))),y)
  0 -> 1            f(h(h(g(x))),h(y))
  1 -> 1            f(h(g(h(g(x)))),y)
  1 -> 1 -> 1       f(h(h(g(g(x)))),y)

  >>> for a in t.apply_([r1,r2], ignore_seen = False, bfs = False):
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

def apply(*a, **kw):
  """Iterate over terms resulting from recursive applications of rules
  to (subterms of) a term."""
  for u in apply_(*a, **kw): yield u.term

def isnormalform(t, rules):
  """Is term a normal form?"""
  return len(list(apply1(t, rules))) == 0

def normalforms(t, rules):                                      # {{{1
  """
  Iterate over normal forms of term (for terminating TRS).

  >>> import trstools as T
  >>> r1  = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> r2  = T.rule("g(h(x))   -> h(g(x))")
  >>> t   = T.term("f(g(h(g(h(x)))),y)")
  >>> for a in t.normalforms([r1,r2]): print(a)
  f(h(h(g(x))),h(y))
  f(h(h(g(g(x)))),y)
  """

  for u in apply(t, rules):
    if isnormalform(u, rules): yield u
                                                                # }}}1

def unify(rules):
  """..."""
  raise "TODO"

def critical_pairs(rules):
  """..."""
  raise "TODO"

# TODO
for f in "subterms_ subterms variables substitute \
          applyrule apply1_ apply1 apply_ apply   \
          isnormalform normalforms".split():
  setattr(Variable, f, vars()[f])
  setattr(Function, f, vars()[f])

if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:]))

# vim: set tw=70 sw=2 sts=2 et fdm=marker :
