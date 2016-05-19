[]: {{{1

    File        : README.md
    Maintainer  : Felix C. Stegerman <flx@obfusk.net>
    Date        : 2016-05-18

    Copyright   : Copyright (C) 2016  Felix C. Stegerman
    Version     : v0.1.1

[]: }}}1

<!-- badge? -->

## Description

trstools.py - python (2+3) term rewriting tools

See `trstools.py` for the code (with examples).

## Examples

```bash
$ ./trstools.py --rule "f(x,h(x)) -> f(x,x)"      \
                --rule "f(g(x),y) -> f(x,h(y))"   \
                --rule "g(h(x))   -> h(g(x)))"    \
                --normalforms "f(g(h(g(h(x)))),y)"
f(h(h(g(x))),h(y))
f(h(h(g(g(x)))),y)

$ cat > /tmp/rules
f(x,h(x)) -> f(x,x)
f(g(x),y) -> f(x,h(y))
g(h(x))   -> h(g(x)))
^D

$ ./trstools.py --rules-from=/tmp/rules --critical-pairs
[ f(x,h(h(g(x)))), f(g(x),g(x)) ]
[ f(h(g(x)),z), f(h(x),h(z)) ]

$ ./trstools.py --rules-from=/tmp/rules --tree "f(g(h(g(h(x)))),y)" \
                                        --mark-nf
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

# open (temporary) graph
$ ./trstools.py --rules-from=/tmp/rules --graph "f(g(h(g(h(x)))),y)"

# save graph ...
$ ./trstools.py --rules-from=/tmp/rules --graph "f(g(h(g(h(x)))),y)" \
                --output /tmp/graph.png
# ... and open
$ xdg-open /tmp/graph.png
```

## TODO

* no double lines in graph?!
* improve, make more efficient?!
* ...

## License

GPLv3+ [1].

## References

[1] GNU General Public License, version 3
--- https://www.gnu.org/licenses/gpl-3.0.html

[]: ! ( vim: set tw=70 sw=2 sts=2 et fdm=marker : )
