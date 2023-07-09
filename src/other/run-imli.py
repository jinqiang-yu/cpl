#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## run-imli.py
##
##

#
#==============================================================================
import collections
import getopt
import os
#import imli
from pyrulelearn import imli, utils
import sys
import gzip
from data import Data
import pandas as pd
import numpy as np
import resource

#
#==============================================================================
def accuracy(y, pred_y):
    n = 0
    assert len(y) == len(pred_y)
    for c, pred_c in zip(y, pred_y):
        if c != pred_c:
            n += 1

    print("\nThe number of errors are: {0} out of {1}".format(n, len(y)))
    print("The accuracy is {0}".format('%0.2f'%((1 - n / len(y)) * 100) + "%" ))


#
#==============================================================================
def bench_print(rule, rtype, feats, target, y_dict):
    assert rtype == 'DNF', 'Only DNF output is supported'

    print('Rules:')

    terms = rule.strip().split('\n')
    terms = list(filter(lambda l: len(l.strip()) > 0, terms))
    default = y_dict.dir[0] if len(terms) > 0 else y_dict.dir[1]

    for term in terms:
        print(term)

    for term in terms:
        term = term.rstrip('OR ')
        term = term.strip(' () ')
        preamble = []
        for lit in term.split(' AND '):
            lit = lit.strip()
            if lit.startswith('is not not'):
                lit = lit[10:].strip()
                val = 1
            elif lit.startswith('is not '):
                lit = lit[7:].strip()
                val = 0
            elif lit.startswith('not ') and not (lit.startswith('not <') or lit.startswith('not >=')):
                lit = lit[4:].strip()
                val = 0
            else:
                #lit = lit[4:].strip()
                lit = lit.strip()
                val = 1
            if lit not in feats:
                assert '"{0}"'.format(lit) in feats, 'Wrong literal'
                lit = '"{0}"'.format(lit)

            preamble.append('\'{0}: {1}\''.format(lit, val))

        print('c2 cover: {0} => {1}: {2}'.format(', '.join(preamble), target, y_dict.dir[1]))

    # default rule
    print('c2 cover: true => {0}: {1}'.format(target, default))
    # print('c2 cover: true => {0}: 0'.format(target))


#
#==============================================================================
def parse_options():
    """
        Parses command-line option
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'abc:hs:t:v',
                ['accy', 'bench' , 'clauses=', 'help', 'solver=', 'type=','verbose'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    test_accy = False
    bench = False
    clauses = 5
    #solver = 'rc2'
    solver = 'open-wbo'
    rtype = 'DNF'
    verbose = 0

    for opt, arg in opts:
        if opt in ('-a', '--accy'):
            test_accy = True
        elif opt in ('-b', '--bench'):
            bench = True
        elif opt in ('-c', '--clauses'):
            clauses = int(arg)
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-s', '--solver'):
            solver = str(arg)
        elif opt in ('-t', '--type'):
            rtype = str(arg).upper()
        elif opt in ('-v', '--verbose'):
            verbose += 1
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return solver, rtype, clauses, bench, test_accy, verbose, args


#
#==============================================================================
def usage():
    """
        Prints usage message.
        """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] dimacs-file')
    print('Options:')
    print('        -a, --accy               Check accuracy')
    print('        -b, --bench              Print the result in the MinDS format')
    print('        -h, --help               Show this help message')
    print('        -c, --clauses=<int>      Number of clauses/terms to represent each class')
    print('                                 Available values: [1 .. INT_MAX] (default: 5)')
    print('        -s, --solver=<string>    MaxSAT solver to use')
    print('                                 Available values: openwbo, maxhs, rc2 (default: rc2)')
    print('        -t, --type=<string>      Rule type to learn')
    print('                                 Available values: cnf, dnf (default: dnf)')
    print('        -v, --verbose            Be verbose')



#
#==============================================================================
if __name__ == '__main__':


    solver, rtype, clauses, bench, test_accy, verbose, files = parse_options()

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime

    if files:
        # initialize the solver
        model = imli.imli(verbose=verbose, solver=solver, num_clause=clauses,
                data_fidelity=20, rule_type=rtype)

        data = Data(filename=files[0])
        data.binarize()
        '''
        print('data.samps:')
        print(data.samps[0])
        print()
        print('data.samps_:')
        print(data.samps_[0])
        print()
        print('data.ffmap:')
        print(data.ffmap)
        print()
        print('data.fvmap:')
        print(data.fvmap)
        '''

        X = []
        y = []
        #for samp in data.samps_:
        for i, samp_ in enumerate(data.samps_):
            row = [1 if s > 0 else 0 for s in samp_[:-1]]
            y_ = samp_[-1]
            wght = data.wghts[i]

            for j in range(wght):
                X.append(row)
                y.append(data.fvmap.opp[y_][-1])
        features = [data.fvmap.opp[col+1] for col in range(len(data.samps_[0])-1)]
        features = list(map(lambda l: '{0} = {1}'.format(l[0], l[1])
                            if '<=' not in l[1] and '<' not in l[1]
                               and '>=' not in l[1] and '>' not in l[1]
                               and '=' not in l[1] else l[1], features))

        y_count = collections.Counter(y)
        sorted_y_unique = sorted(set(y), key=lambda l: y_count[l], reverse=True)

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

        y = list(map(lambda l: y_dict.opp[l], y))
        y = np.array(y)
        target = data.names[-1]

        # hopefully, this should not discretize anything
        # as the datasets we target are already binary
        #X, y, features = utils.discretize_orange(files[0])

        print('Fitting...')
        # learn a classifier
        model.fit(X, y)

        print('Getting Rules...')
        # get the result
        rule = model.get_rule(features)

        print('Rules computed')

        if test_accy:
            print(rule)
            print(model.get_selected_column_index())
            print(model.get_threshold_clause())
            print(model.get_threshold_literal())

            pred_y = model.predict(X, y)
            print("\nFor training datasets:")
            accuracy(y, pred_y)

            if len(files) > 1:
                test_X, test_y, test_features = model.discretize(files[1])
                pred_test_y = model.predict(test_X, test_y, True)
                print("\nFor testing datasets:")
                accuracy(test_y, pred_test_y)
        elif bench:
            bench_print(rule, rtype, feats=features, target=target, y_dict=y_dict)

    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

    print('c2 rtime: {0}'.format(time))