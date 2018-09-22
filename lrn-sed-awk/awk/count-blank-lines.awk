# count blank lines

/^$/ { print x++ }
END {print "--- " x}
