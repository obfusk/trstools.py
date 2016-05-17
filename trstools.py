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
    self.name = name
    self.args = args

  def __repr__(self):
    return self.name + "(" + ",".join(map(repr, self.args)) + ")"

  def __eq__(self, rhs):
    if not isinstance(rhs, Function): return False
    return self.name == rhs.name and self.args == rhs.args
                                                                # }}}1

class Variable(object):                                         # {{{1
  """..."""

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name

  def __eq__(self, rhs):
    if not isinstance(rhs, Variable): return False
    return self.name == rhs.name
                                                                # }}}1

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

def term(parseresult):
  """Turns parse result into a nested Function/Variable tree."""
  if "varname" in parseresult:
    return Variable(parseresult.varname)
  return Function(parseresult.funcname,
                  *map(term, parseresult.subterms))

def parse_term(t, fun = FUNCTIONS, var = VARIABLES):            # {{{1
  r"""
  Parses a term; returns nested Function/Variable tree.

  >>> import trstools as T
  >>> t1  = T.parse_term("f(g(h(x)),y)")
  >>> f,v = T.Function, T.Variable
  >>> t2  = f("f",f("g",f("h",v("x"))),v("y"))
  >>> t1 == t2
  True
  """

  lp, rp  = P.Literal("("), P.Literal(")")
  fu, va  = fun("funcname"), var("varname")
  expr    = P.Forward()
  st      = P.delimitedList(P.Group(expr), ",")
  expr << ( va | fu + lp + P.Optional(st("subterms")) + rp )
  return term(expr.parseString(t))
                                                                # }}}1

def rule(l, r = None):
  """..."""
  if isinstance(l, rule):
    return l
  if r is None:
    l, r = l.split("->")
  return Rule(parse_term(l), parse_term(r))

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
