#!/usr/bin/python

# --                                                            ; {{{1
#
# File        : trstools.py
# Maintainer  : Felix C. Stegerman <flx@obfusk.net>
# Date        : 2016-05-17
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

def main(*args):                                                # {{{1
  p = argument_parser(); n = p.parse_args(args)
  if n.test:
    import doctest
    doctest.testmod(verbose = n.verbose)
    return 0
  # ... TODO ...
  return 0
                                                                # }}}1

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

FUNCTIONS = P.Word("fgh", P.alphas + "'")
VARIABLES = P.Word("xyz", P.alphas + "'")

# TODO: improve
class Function(object):                                         # {{{1
  """Function with zero-or-more arguments."""

  def __init__(self, name, *args):
    self.name = name; self.args = list(args)

  def copy(self):
    return type(self)(self.name, *[ a.copy() for a in self.args ])

  def __repr__(self):
    return self.name + "(" + ",".join(map(repr, self.args)) + ")"

  def __eq__(self, rhs):
    if not isfunc(rhs): return False
    return self.name == rhs.name and self.args == rhs.args
                                                                # }}}1

def isfunc(x): return isinstance(x, Function)

# TODO: improve
class Variable(object):                                         # {{{1
  """Variable."""

  def __init__(self, name):
    self.name = name

  def copy(self):
    return type(self)(self.name)

  def __repr__(self):
    return self.name

  def __eq__(self, rhs):
    if not isvar(rhs): return False
    return self.name == rhs.name
                                                                # }}}1

def isvar(x): return isinstance(x, Variable)

class Rule(object):                                             # {{{1
  """Rule (left maps to right)."""

  def __init__(self, l, r):
    self._l = l; self._r = r

  @property
  def left(self):
    return self._l

  @property
  def right(self):
    return self._r

  def __repr__(self):
    return repr(self._l) + " -> " + repr(self._r)
                                                                # }}}1

class Ruleset(object):                                          # {{{1
  """Set of rules."""

  def __init__(*rules):
    self._rules = tuple(map(rule, rules))

  @property
  def rules(self):
    return self._rules
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

  if isfunc(x) or isvar(x): return x
  elif isinstance(x, P.ParseResults):
    if "varname" in x: return Variable(x.varname)
    return Function(x.funcname, *map(term, x.subterms))
  else: return term(parse_term(x))
                                                                # }}}1

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

# TODO: improve
def subterms_(t, proper = False, variable = True):              # {{{1
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
    def h(x): u = t.copy(); u.args[i] = f(x); return u
    return h
  if isfunc(t):
    if not proper:
      yield Subterm(t, lambda x: x)
    for i, u in enumerate(t.args):
      for v, f in subterms_(u, variable = variable):
        yield Subterm(v, g(i, f))
  elif variable and not proper:
    yield Subterm(t, lambda x: x)
                                                                # }}}1

def subterms(t, proper = False, variable = True):
  """Iterate over subterms of a term."""
  for u in subterms_(t, proper, variable): yield u.term

#TODO: improve
def substitute_(t, sigma):                                      # {{{1
  r"""
  Substitute terms for variables.

  NB: modifies the term (or returns a new one if needed)!

  >>> import trstools as T
  >>> sigma = dict(x = "z", y = "g(x)")
  >>> T.term("f(g(h(x)),y)").substitute(sigma)
  f(g(h(z)),g(x))
  """

  if isvar(t): return term(sigma.get(t.name, t))
  t.args = [ substitute(u, sigma) for u in t.args ]
  return t
                                                                # }}}1

def substitute(t, sigma):
  """Substitute terms for variables (in a copy of the term)."""
  return substitute_(t.copy(), sigma)

# TODO: improve
def applyrule_(t, r, variables = None):                         # {{{1
  """
  Apply a rule to a term (if possible).

  NB: modifies the term (or returns a new one if needed)!

  >>> import trstools as T
  >>> r = T.rule("f(g(x),y) -> f(x,h(y))")
  >>> T.term("f(g(h(x)),y)").applyrule(r)
  f(h(x),h(y))
  """

  lhs, rhs = r.left, r.right
  if variables is None: variables = {}
  if isvar(lhs):
    if lhs.name not in variables or variables[lhs.name] == t:
      variables[lhs.name] = t
      return t
  elif isfunc(t):
    if t.name == lhs.name:
      if len(t.args) != len(lhs.args):
        raise("functions differ in arity!")                     # TODO
      for i, u in enumerate(t.args):
        r_ = Rule(lhs.args[i], None)
        if not applyrule_(u, r_, variables): return None
      return substitute_(rhs, variables) if rhs is not None else True
  return None
                                                                # }}}1

def applyrule(t, rule):
  """Apply a rule to (a copy of a) term (if possible)."""
  return applyrule_(t.copy(), rule)

def apply1():
  """..."""
  raise "TODO"

def apply(n = None):
  """..."""
  raise "TODO"

def normalforms():
  """..."""
  raise "TODO"

def unify():
  """..."""
  raise "TODO"

# TODO: more...
for f in "subterms_ subterms substitute_ substitute \
          applyrule_ applyrule".split():
  setattr(Variable, f, vars()[f])
  setattr(Function, f, vars()[f])

if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:]))

# vim: set tw=70 sw=2 sts=2 et fdm=marker :
