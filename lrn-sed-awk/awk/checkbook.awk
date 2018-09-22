# checkbook.awk
BEGIN {FS = "\t"; OFS = "\t"}

#1 Expect the first record to have the starting balance.

NR == 1 {print "Beginning Balance \t" $1
         balance = $1
         next           # get next record and start over
}

#2 Apply to each check record, adding amount from balance.
{
    print $1, $2, $3
    balance += $3   # checks have negative amounts
    print "current balance " balance
}
