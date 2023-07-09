#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## dsplit.py
##
##

#
#==============================================================================
from __future__ import print_function
import collections
import getopt
import math
import os
import random
import sys


#
#==============================================================================
class Data(object):
    """
        A class for reading and storing a dataset.
    """

    def __init__(self, from_fp=None, separator=None):
        """
            Constructor.
        """

        self.names = None
        self.feats = None
        self.samps = None

        if from_fp:
            self.from_fp(fp=from_fp, separator=separator)

    def from_fp(self, fp=None, separator=','):
        """
            Get data from a CSV file.
        """

        if fp:
            lines = fp.readlines()

            # reading preamble
            self.names = lines[0].strip().split(separator)
            self.feats = [set([]) for n in self.names]
            del(lines[0])

            # filling name to id mapping
            self.nm2id = {name: i for i, name in enumerate(self.names)}

            # reading training samples
            self.samps, self.classes = [], {}

            for line in lines:
                sample = line.strip().split(separator)

                for i, f in enumerate(sample):
                    if f:
                        self.feats[i].add(f)

                self.samps.append(sample)

                # clasterizing the samples
                if sample[-1] not in self.classes:
                    self.classes[sample[-1]] = []

                self.classes[sample[-1]].append(len(self.samps) - 1)

    def subsample(self, perc, rseed):
        """
            Subsample a given dataset.
        """

        # printing preamble into training and test datasets
        print(','.join(self.names), file=sys.stdout)
        print(','.join(self.names), file=sys.stderr)

        # seeding the random number generator
        if rseed:
            random.seed(rseed)
        else:
            random.seed()

        # do the splitting
        for lb in self.classes:
            random.shuffle(self.classes[lb])

            # we are taking perc% for the training data
            nof_training = int(math.floor(len(self.classes[lb]) * perc))

            # printing the training data to stdout
            for i in range(nof_training):
                print(','.join(self.samps[self.classes[lb][i]]), file=sys.stdout)

            # printing the test data to stderr
            for i in range(nof_training, len(self.classes[lb])):
                print(','.join(self.samps[self.classes[lb][i]]), file=sys.stderr)


#
#==============================================================================
def parse_options():
    """
        Parses command-line options.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'hp:r:s:',
                                   ['help',
                                    'perc=',
                                    'rseed=',
                                    'sep='])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize() + '\n')
        usage()
        sys.exit(1)

    perc = 0.8
    rseed = None
    sep = ','

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-p', '--perc'):
            perc = float(arg)
        elif opt in ('-r', '--rseed'):
            rseed = None if str(arg) == 'none' else int(arg)
        elif opt in ('-s', '--sep'):
            sep = str(arg)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return perc, rseed, sep, args


#
#==============================================================================
def usage():
    """
        Prints help message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] file')
    print('Options:')
    print('        -h, --help             Show this message')
    print('        -p, --perc=<float>     Percentage of the samples to use')
    print('                               Available values: (0 .. 1] (default = 0.1)')
    print('        -r, --rseed=<int>      Seed for the random number generator')
    print('                               Available values: [0 .. INT_MAX], none (default = none)')
    print('        -s, --sep=<string>     Separator used in the input CSV file (default = \',\')')


#
#==============================================================================
if __name__ == '__main__':
    perc, rseed, sep, files = parse_options()

    if files:
        with open(files[0]) as fp:
            data = Data(from_fp=fp, separator=sep)
    else:
        data = Data(from_fp=sys.stdin, separator=sep)

    data.subsample(perc, rseed)
