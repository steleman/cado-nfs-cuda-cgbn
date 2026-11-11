# This is a buggy strategy file
blah = { ECM-B12(12345) 200 2000 }
foo = { ECM-B12 200 2000 }
# foo after blah should fail, because a default parameter choice cannot
# work after a non-default one.
10 20: { blah foo }
