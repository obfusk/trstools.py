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

import argparse, pyparsing as P, sys

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

class Function(object):                                         # {{{1
  """..."""

  def __init__(self, name, *args):
    self.name = name; self.args = args

  def __repr__(self):
    return self.name + "(" + ",".join(map(repr, self.args)) + ")"

  def __eq__(self, rhs):
    if not isfunc(rhs): return False
    return self.name == rhs.name and self.args == rhs.args
                                                                # }}}1

def isfunc(x): return isinstance(x, Function)

class Variable(object):                                         # {{{1
  """..."""

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name

  def __eq__(self, rhs):
    if not isvar(rhs): return False
    return self.name == rhs.name
                                                                # }}}1

def isvar(x): return isinstance(x, Variable)

class Rule(object):                                             # {{{1
  """..."""

  def __init__(self, l, r):
    self._l = l; self._r = r

  @property
  def left(self):
    return self.l

  @property
  def right(self):
    return self.r
                                                                # }}}1

class Ruleset(object):                                          # {{{1
  """..."""

  def __init__(*rules):
    self._rules = tuple(map(rule, rules))

  @property
  def rules(self):
    return self._rules
                                                                # }}}1

def term(x):                                                    # {{{1
  """Turns string or parse result into a nested Function/Variable
  tree."""

  if isfunc(x) or isvar(x): return x
  elif isinstance(x, P.ParseResults):
    if "varname" in x: return Variable(x.varname)
    return Function(x.funcname, *map(term, x.subterms))
  else: return term(parse_term(x))
                                                                # }}}1

def parse_term(t, func = FUNCTIONS, var = VARIABLES):           # {{{1
  r"""
  Parses a term.

  >>> import trstools as T
  >>> t1  = T.term("f(g(h(x)),y)")
  >>> f,v = T.Function, T.Variable
  >>> t2  = f("f",f("g",f("h",v("x"))),v("y"))
  >>> t1 == t2
  True
  """

  lp, rp  = P.Literal("("), P.Literal(")")
  fu, va  = func("funcname"), var("varname")
  expr    = P.Forward()
  st      = P.delimitedList(P.Group(expr), ",")
  expr << ( va | fu + lp + P.Optional(st("subterms")) + rp )
  return expr.parseString(t)
                                                                # }}}1

def rule(l, r = None):
  """..."""
  if isinstance(l, Rule): return l
  if r is None: l, r = l.split("->")
  return Rule(term(l), term(r))

def ruleset():
  """..."""

def subterms():
  """..."""

def substitute():
  """..."""

def apply1():
  """..."""

def apply(n = None):
  """..."""

def normalforms():
  """..."""

def unify():
  """..."""

if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:]))

# vim: set tw=70 sw=2 sts=2 et fdm=marker :
