#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## ripper2.py
##
##

#
#==============================================================================
import collections
import getopt
import gzip
import os
import pandas as pd
import sys
import wittgenstein as lw
import resource
import data

#
#==============================================================================
def bench_print(ruleset, feats, target, default, y_dict):
    """
        Prints the rule set in the right format.
    """

    fmap = {f.replace(' ', ''): f for f in feats}

    ruleset = str(ruleset)
    print(feats)
    for term in ruleset.strip().split(' V '):
        term = term.strip(' [] ')

        preamble = []
        for lit in term.split('^'):
            print(lit)
            lit = lit.strip()
            lit, val = lit.rsplit('=', 1)
            if lit in fmap:
                lit = fmap[lit]

            elif lit not in feats:
                assert '"{0}"'.format(lit) in feats, 'Wrong literal {0}'.format(lit)
                lit = '"{0}"'.format(lit)

            preamble.append('\'{0}: {1}\''.format(lit, val))

        print('c2 cover: {0} => {1}: {2}'.format(', '.join(preamble), target, y_dict.dir[1]))

    # default rule
    print('c2 cover: true => {0}: {1}'.format(target, default))


#
#==============================================================================
def parse_options():
    """
        Parses command-line options.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h', ['help'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return args


#
#==============================================================================
def usage():
    """
        Prints help message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] csv-file')
    print('Options:')
    print('        -h, --help    Print this usage message')


#
#==============================================================================
if __name__ == "__main__":
    files = parse_options()

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime

    if files:

        df = pd.read_csv(files[0])
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        features = df.columns[:-1]

        y_count = collections.Counter(y)
        sorted_y_unique = sorted(y.unique(), key=lambda l: y_count[l], reverse=True)

        YIDMap = collections.namedtuple('YIDMap', ['dir', 'opp'])
        y_dict = YIDMap(dir={}, opp={})
        for i, y_value in enumerate(sorted_y_unique):
            if i == 0:
                yid = 1
            elif i == 1:
                yid = 0
            else:
                yid = i
            y_dict.dir[yid] = y_value
            y_dict.opp[y_value] = yid

        feats = df.columns.values[:-1]
        target = df.columns.values[-1]
        default = y_dict.dir[0]
        df[target] = df[target].map(y_dict.opp)
        

        # running RIPPER
        ripper_clf = lw.RIPPER()
        ripper_clf.fit(df, class_feat=target, pos_class=1)

        bench_print(ripper_clf.ruleset_, feats, target, default, y_dict)

    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

    print('c2 rtime: {0}'.format(time))
