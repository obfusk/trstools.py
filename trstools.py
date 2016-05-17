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

import argparse, pyparsing, sys

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

class Function(object):                                         # {{{1
  """..."""

  def __init__(self, name):
    self.name = name

  def __str__(self):
    return self.name

  def __repr__(self):
    return "<fun>" + self.name
                                                                # }}}1

class Variable(object):                                         # {{{1
  """..."""

  def __init__(self, name):
    self.name = name

  def __str__(self):
    return self.name

  def __repr__(self):
    return "<var>" + self.name

                                                                # }}}1

class Term(object):                                             # {{{1
  """..."""

  def __init__():
    pass

                                                                # }}}1

class Rule(object):                                             # {{{1
  """..."""

  def __init__():
    pass

                                                                # }}}1

class Ruleset(object):                                          # {{{1
  """..."""

  def __init__():
    pass

                                                                # }}}1

def parse_term():
  """..."""

def rule():
  """..."""

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
