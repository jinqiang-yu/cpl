#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## run-pyids.py (should be used only with Python 3)
##
##

#
#==============================================================================
from __future__ import print_function
import getopt
import gzip
from io import StringIO
import os
from six.moves import range
import sys
import tempfile
import resource

import pandas as pd
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent
from pyarc.qcba.data_structures import QuantitativeDataFrame
import io


#
#==============================================================================
def dump_result(rules, default, auc): #data, clf, learner, to=sys.stdout):
    """
        Print resulting (unordered) list of rules to a file pointer.
    """

    nof_lits = 1  # count the default branch as well
    for rule in rules:
        conds, target = str(rule).split('=>')
        conds = conds.split('IDSRule', maxsplit=1)[-1].rsplit('}', maxsplit=1)[0].strip().strip('{}').split(',')
        conds_ = []
        nof_lits += len(conds)
        for cond in conds:
            cond = cond.rsplit('=', maxsplit=1)
            if cond[-1] == '1':
                cond[0] = "not '" + cond[0]
                cond[1] = "0'"
            else:
                cond[0] = "'" + cond[0]
                cond[1] = cond[1] + "'"
            cond = ': '.join(cond)
            conds_.append(cond)
        conds = ', '.join(conds_)
        target = target.rsplit('}', maxsplit=1)[0].strip().strip('{}').replace('=', ': ')
        rule = '{} => {}'.format(conds, target)
        print("c* cover: {}".format(rule))
    print("c* cover: true => {}".format(default))
    print("c* cover size: {}".format(len(rules) + 1))
    print("c* cover wght: {}".format(nof_lits))
    print("c* cover auc: {}".format(auc))

#
#==============================================================================
def parse_options():
    """
        Parses command-line options.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'hlos:',
                                   ['help',
                                    'list',
                                    'orange-fmt',
                                    'sep='])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize() + '\n')
        usage()
        sys.exit(1)

    dlist = False
    orange_fmt = False
    sep = ','

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-l', '--list'):
            dlist = True
        elif opt in ('-o', '--orange-fmt'):
            orange_fmt = True
        elif opt in ('-s', '--sep'):
            sep = str(arg)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return dlist, orange_fmt, sep, args


#
#==============================================================================
def usage():
    """
        Prints help message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] file')
    print('Options:')
    print('        -h, --help')
    print('        -s, --sep=<string>    Separator used in the input CSV file (default = \',\')')


#
#==============================================================================
if __name__ == '__main__':
    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime

    # assert sys.version_info.major == 3, 'Python2 is not supported'
    dlist, orange_fmt, sep, files = parse_options()

    if files:
        df = pd.read_csv(files[0])

        quant_df = QuantitativeDataFrame(df)
        cars = mine_CARs(df, 20)

    #cars = mine_CARs(df, rule_cutoff=50)
    #lambda_array = [1, 1, 1, 1, 1, 1, 1]

    #quant_dataframe = QuantitativeDataFrame(df)

    #ids = IDS(algorithm="SLS")
    #ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

    #acc = ids.score(quant_dataframe)
    #print('acc:', acc)
    #exit()

    def fmax(lambda_dict):
        #print('lambda_dict:', lambda_dict)
        c_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime
        ids = IDS(algorithm="SLS")
        ids.fit(class_association_rules=cars, quant_dataframe=quant_df, lambda_array=list(lambda_dict.values()))
        default = '{}: {}'.format(df.columns[-1], ids.clf.default_class)
        auc = ids.score_auc(quant_df)
        dump_result(ids.clf.rules, default, auc)

        c_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                 resource.getrusage(resource.RUSAGE_SELF).ru_utime - c_time
        print('c* rtime: {0}'.format(c_time))
        return auc

    coord_asc = CoordinateAscent(
        func=fmax,
        func_args_ranges=dict(
            l1=(1, 1000),
            l2=(1, 1000),
            l3=(1, 1000),
            l4=(1, 1000),
            l5=(1, 1000),
            l6=(1, 1000),
            l7=(1, 1000)
        ),
        ternary_search_precision=50,
        max_iterations=3
    )

    best_lambdas = coord_asc.fit()
    print('best_lambdas:', best_lambdas)
    print(best_lambdas)

    print('\nfinal ds:')
    ids = IDS(algorithm="SLS")
    ids.fit(class_association_rules=cars, quant_dataframe=quant_df, lambda_array=best_lambdas)
    default = '{}: {}'.format(df.columns[-1], ids.clf.default_class)
    auc = ids.score_auc(quant_df)
    dump_result(ids.clf.rules, default, auc)

    f_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

    print('c* f_rtime: {0}'.format(f_time))
    exit()
