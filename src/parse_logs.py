#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
from data import Data
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset, discretize_dataset
from xgbooster.tree import TreeEnsemble
import pandas as pd
import numpy as np
import six
import glob
import collections
import subprocess
import json
import collections
import pickle
import statistics
import itertools

default = True

class DS(object):
    def __init__(self, ds_log):
        self.nof_lits = None
        self.nof_rules = None
        self.tot_size = None
        self.default = None
        self.init_(ds_log)
    def __str__(self):
        return '\n'.join(self.lines_str)

    def init_(self, ds_log):
        self.rules = collections.defaultdict(lambda : [])
        with open(ds_log, 'r') as f:
            lines = f.readlines()

        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            if ' rtime:' in line or 'c3 total time:' or 'c* f_rtime:' in line:
                self.rtime = float(line.split(':', maxsplit=1)[-1])
                break

        if '/ids/' in ds_log:
            for i, line in enumerate(lines):
                if 'final ds:' in line:
                    lines = lines[i+1:]
                    break

        lines = list(filter(lambda l: 'cover:' in l, lines))
        lines = list(map(lambda l: l.split('cover:', maxsplit=1)[-1].strip(), lines))
        self.lines_str = lines[:]
        #print('\n'.join(lines))
        for line in lines:
            cond, pred = line.split(' => ')
            pred = pred.split(': ', maxsplit=1)[-1].strip("'").strip()

            if cond.strip().strip("'").strip('"').strip().lower() == 'true':
                self.default = pred
                continue

            #try:
            #    pred = xgb.target_name[int(float(pred))]
            #except:
            #    pass
            #print(line)
            #print(pred)
            #print(cond)
            cond = cond.split("', ")
            cond = list(map(lambda l: l.strip("'"), cond))
            #print(cond)
            condition = []
            features = set()
            for fv in cond:
                #print(fv)
                if fv.startswith("not ") or fv.startswith("NOT "):
                    sign = False
                else:
                    sign = True
                feature, value = fv.split(': ', maxsplit=1)
                value = value.strip()
                if not sign:
                    if fv.startswith("not "):
                        feature = feature.split("not '", maxsplit=1)[-1]
                    else:
                        feature = feature.split("NOT ", maxsplit=1)[-1]

                eq2 = None
                if '==' in feature:
                    eq = '=='
                elif '!=' in feature:
                    eq = '!='
                elif '>' in feature:
                    if feature.count('>') == 2:
                        if '>=' not in feature:
                            eq, eq2 = '>'
                        elif feature.count('>=') == 2:
                            eq, eq2 = '>='
                        else:
                            if feature.index('>=') > feature.index('>'):
                                eq = '>'
                                eq2 = '>='
                            else:
                                eq = '>='
                                eq2 = '>'
                    else:
                        eq = '>' if '>=' not in feature else '>='
                elif '<' in feature:
                    if feature.count('<') == 2:
                        if '<=' not in feature:
                            eq, eq2 = '<'
                        elif feature.count('<=') == 2:
                            eq, eq2 = '<='
                        else:
                            if feature.index('<=') > feature.index('<'):
                                eq = '<'
                                eq2 = '<='
                            else:
                                eq = '<='
                                eq2 = '<'
                    else:
                        eq = '<' if '<=' not in feature else '<='
                elif '=' in feature:
                    eq = '=='
                    feature = feature.replace('=', '==')
                else:
                    eq = '='

                value2 = None
                if eq != '=':
                    if value.lower() in ('0', 'false'):
                        sign = not sign
                    if eq2 is None:
                        feature, value = feature.split(eq, maxsplit=1)
                        if eq == '!=':
                            eq = '='
                            sign = not sign
                        feature = feature.strip()
                        value = value.strip()
                        if eq == '==':
                            eq = '='
                    else:
                        value, feature = feature.split(eq, maxsplit=1)
                        value = value.strip()
                        feature, value2 = feature.rsplit(eq2, maxsplit=1)
                        feature = feature.strip()
                        value2 = value2.strip()

                feature = feature.strip().strip('"').strip("'").strip()
                condition.append({'feature': feature,
                                   'value': value,
                                  'value2': value2,
                                   'eq': eq,
                                   'eq2': eq2,
                                   'sign': sign})
                features.add(feature)
                #print('feature:', feature)
                #print('eq:',eq)
                #print('value:', value)
                #print('eq2:',eq2)
                #print('value2:', value2)
                #print(sign)
                #print()
            self.rules[pred].append({'condition': condition, 'features': features}) #print(self.rules)

    def add_default(self, train):
        if self.default is None:
            self.default = train[train.columns[-1]].mode()[0]

    def accry_overlap(self, df):
        #print(df)
        correct = []
        self.overlap = 0
        self.xp_size = []
        self.count_tot_size()
        for i in range(df.shape[0]):
            pred2size = collections.defaultdict(lambda : [])
            inst = df.iloc[i]
            label = inst.iloc[-1]
            for pred in self.rules:
                rules = self.rules[pred]
                for rule_feats in rules:
                    rule = rule_feats['condition']
                    lits = len(rule_feats['features'])
                    size = lits + 1
                    if self.match(inst, rule):
                        pred2size[pred].append(size)

            if len(pred2size) == 0 and self.default is not None:
                pred2size[self.default].append(self.count_tot_size())

            if len(pred2size) == 1 and self.cmpr('=', df.columns[-1], label, list(pred2size.keys())[0]) :
                correct.append(True)
                self.xp_size.append(statistics.mean(list(pred2size.values())[0]))
            else:
                #print('wrong predictions')
                #print(pred2size.keys())
                correct.append(False)

                if len(pred2size) > 1:
                    self.overlap += 1

                if len(pred2size) >= 1:
                    self.xp_size.append(sum([statistics.mean(pred2size[pred]) for pred in pred2size]))
                else:
                    self.xp_size.append(self.count_tot_size())

        self.acry = round(sum(correct) / len(correct) * 100, 2)
        self.overlap_rate = round(self.overlap / len(correct) * 100, 2)
        self.xp_size = round(sum(self.xp_size) / len(self.xp_size), 2)

        return self.acry

    def match(self, inst, rule):
        for cond in rule:
            #print(cond)
            feature = cond['feature'].strip("'")
            inst_val = inst[feature]
            #print('inst_val:', inst_val)
            eq = cond['eq']
            cond_val = cond['value']
            eq2 = cond['eq2']
            cond_val2 = cond['value2']
            sign = cond['sign']

            cmpr = self.cmpr(eq, feature, inst_val, cond_val, eq2, cond_val2)
            if (not cmpr and sign) or (cmpr and not sign):
                #print('not matched\n')
                return False
            else:
                #print('matched\n')
                pass

        return True

    def cmpr(self, eq, feature, inst_val, cond_val, eq2=None, cond_val2=None):
        #print('eq: {}, feature: {}, inst_val: {}, cond_val: {}, eq2: {}, cond_val2: {}' \
        #      .format(eq, feature, inst_val, cond_val, eq2, cond_val2))

        match = False
        if eq == '=':
            try:
                if float(inst_val) == float(cond_val):
                    match = True
            except:
                if str(inst_val).lower() == str(cond_val).lower():
                    match = True
                elif str(inst_val).lower().replace(' ', '') == '{0}={1}'.format(feature, cond_val).strip().lower().replace(' ', ''):
                    match = True
                elif str(inst_val).lower().replace(' ', '') == '{0}=={1}'.format(feature, cond_val).strip().lower().replace(' ', ''):
                    match = True
        elif eq == '<':
            if eq2 is None:
                try:
                    if float(inst_val) < float(cond_val):
                        match = True
                except:
                    if str(inst_val).lower().replace(' ', '') == '{0}<{1}'.format(feature, cond_val).strip().lower().replace(' ', ''):
                        match = True
            else:
                if eq2 == '<':
                    try:
                        if float(cond_val) < float(inst_val) < float(cond_val2):
                            match = True
                    except:
                        if str(inst_val).lower().replace(' ', '') == '{0}<{1}<{2}'.format(cond_val,
                                                                                          feature,
                                                                                          cond_val2).strip().lower().replace(' ', ''):
                            match = True
                else:
                    try:
                        if float(cond_val) < float(inst_val) <= float(cond_val2):
                            match = True
                    except:
                        if str(inst_val).lower().replace(' ', '') == '{0}<{1}<={2}'.format(cond_val,
                                                                                          feature,
                                                                                          cond_val2).strip().lower().replace(' ', ''):
                            match = True


        elif eq == '<=':
            if eq2 is None:
                try:
                    if float(inst_val) <= float(cond_val):
                        match = True
                except:
                    if str(inst_val).lower().replace(' ', '') == '{0}<={1}'.format(feature, cond_val).strip().lower().replace(' ', ''):
                        match = True
            else:
                if eq2 == '<':
                    try:
                        if float(cond_val) <= float(inst_val) < float(cond_val2):
                            match = True
                    except:
                        if str(inst_val).lower().replace(' ', '') == '{0}<={1}<{2}'.format(cond_val,
                                                                                          feature,
                                                                                          cond_val2).strip().lower().replace(' ', ''):
                            match = True
                else:
                    try:
                        if float(cond_val) <= float(inst_val) <= float(cond_val2):
                            match = True
                    except:
                        if str(inst_val).lower().replace(' ', '') == '{0}<={1}<={2}'.format(cond_val,
                                                                                          feature,
                                                                                          cond_val2).strip().lower().replace(' ', ''):
                            match = True
        elif eq == '>=':
            if eq2 is None:
                try:
                    if float(inst_val) >= float(cond_val):
                        match = True
                except:
                    if str(inst_val).lower().replace(' ', '') == '{0}>={1}'.format(feature, cond_val).strip().lower().replace(' ', ''):
                        match = True
            else:
                if eq2 == '>':
                    try:
                        if float(cond_val) >= float(inst_val) > float(cond_val2):
                            match = True
                    except:
                        if str(inst_val).lower().replace(' ', '') == '{0}>={1}>{2}'.format(cond_val,
                                                                                           feature,
                                                                                           cond_val2).strip().lower().replace(
                            ' ', ''):
                            match = True
                else:
                    try:
                        if float(cond_val) >= float(inst_val) >= float(cond_val2):
                            match = True
                    except:
                        if str(inst_val).lower().replace(' ', '') == '{0}>={1}>={2}'.format(cond_val,
                                                                                           feature,
                                                                                           cond_val2).strip().lower().replace(
                            ' ', ''):
                            match = True

        else:
            assert eq == '>'

            if eq2 is None:
                try:
                    if float(inst_val) > float(cond_val):
                        match = True
                except:
                    if str(inst_val).lower().replace(' ', '') == '{0}>{1}'.format(feature,
                                                                                   cond_val).strip().lower().replace(' ',
                                                                                                                     ''):
                        match = True
            else:
                if eq2 == '>':
                    try:
                        if float(cond_val) > float(inst_val) > float(cond_val2):
                            match = True
                    except:
                        if str(inst_val).lower().replace(' ', '') == '{0}>{1}>{2}'.format(cond_val,
                                                                                           feature,
                                                                                           cond_val2).strip().lower().replace(
                            ' ', ''):
                            match = True
                else:
                    try:
                        if float(cond_val) > float(inst_val) >= float(cond_val2):
                            match = True
                    except:
                        if str(inst_val).lower().replace(' ', '') == '{0}>{1}>={2}'.format(cond_val,
                                                                                           feature,
                                                                                           cond_val2).strip().lower().replace(
                            ' ', ''):
                            match = True
        return match

    def count_lits(self):
        # may need change if a feature is used more than one in a rule
        if self.nof_lits:
            return self.nof_lits

        self.nof_lits = 0
        for pred in self.rules:
            rules = self.rules[pred]
            for rule_feats in rules:
                features = rule_feats['features']
                self.nof_lits += len(features)

        #if self.default is not None:
        #    self.nof_lits += 1

        return self.nof_lits

    def count_rules(self):
        if self.nof_rules:
            return self.nof_rules

        self.nof_rules = 0
        for pred in self.rules:
            rules = self.rules[pred]
            self.nof_rules += len(rules)

        if self.default is not None:
            self.nof_rules += 1

        return self.nof_rules

    def count_tot_size(self):
        if self.tot_size:
            return self.tot_size

        self.tot_size = self.count_lits() + self.count_rules()

        return self.tot_size

def cmptmax(files, key):
    maxkey = 0
    for file in files:
        with open(file, 'r') as f:
            keydict = json.load(f)

        for dt in keydict['stats']:
            k = keydict['stats'][dt][key]
            if k > maxkey:
                maxkey = k
    return maxkey

def log_check(log, cmpl):
    with open(log, 'r') as f:
        lines = f.readlines()

    identity = 'rtime' if cmpl else 'c3 total time: '

    if 'ripper' in log or 'cn2' in log \
        or 'imli' in log:
        identity = 'rtime:'

    if '/ids/' in log:
        identity = 'c* f_rtime:'

    rtime_lines = list(filter(lambda l: identity in l, lines))
    status = len(rtime_lines) > 0

    timeout_lines = list(filter(lambda l: 'Compilation time out' in l, lines))
    timeout = len(timeout_lines) > 0

    return status, timeout

def parse_dslog(logs, conf, default):
    if 'cmpl' in conf:
        label = 'cpl'
        conf_ = ''
        if 'train' not in conf_:
            conf_ += 'f'
        if conf.endswith('ra'):
            conf_ += 'l'
        elif conf.endswith('ra0.005'):
                conf_ += 'l\lambda_1'
        elif conf.endswith('ra0.05'):
                conf_ += 'l\lambda_2'
        elif conf.endswith('ra0.5'):
                conf_ += 'l\lambda_3'

        if 'rrule' in conf:
            conf_ += 'r'

        if len(conf_) > 0:
            label = label + '$_{' + conf_ + '}$'

    elif 'imli' in conf:
        label = 'imli$_{' + conf[4:] + '}$'

    elif 'sp0.' in conf or 'cmplrb' in conf:
        label = 'sp' if 'sp0.' in conf else 'cmplrb'
        if '0.005' in conf:
            if 'ceil' not in conf:
                label += '$_{\lambda_1}x$'
            else:
                label += '$_{\lambda_1}$'
        elif '0.05' in conf:
            if 'ceil' not in conf:
                label += '$_{\lambda_2}x$'
            else:
                label += '$_{\lambda_2}$'
        elif '0.5' in conf:
            if 'ceil' not in conf:
                label += '$_{\lambda_3}x$'
            else:
                label += '$_{\lambda_3}$'
    else:
        label = conf

    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    cmpl = True if 'cmpl' in conf else False
    for log in logs:
        status, timeout = log_check(log, cmpl)
        if '_train' in log:
            dtname = '_'.join(log.rsplit('/', maxsplit=1)[-1].split('_', maxsplit=2)[:2])
        else:
            dtname = log.rsplit('/', maxsplit=1)[-1].split('_', maxsplit=1)[0]
        dtname = dtname.replace('.log', '')
        if (not status):
            info['stats'][dtname] = {'status': False,
                                     'status2': False,
                                     'rtime': 3600,
                                     'acry': 0.00,
                                     'overlaprate': 100.00,
                                     'rules': 0,
                                     'lits': 0,
                                     'totsize': 0,
                                     'explsize': 0}
            continue

        status2 = not timeout
        ds = DS(log)

        test_path = '../datasets/test/{0}/{1}.csv'.format( \
            log.rsplit('/', maxsplit=1)[-1].split('_', maxsplit=1)[0], \
            dtname)
        test_path = test_path.replace('_train', '_test')
        test = pd.read_csv(test_path)
        test.columns = list(map(lambda l: l.strip("'").strip(), test.columns))

        if ds.default is None:
            train_path = test_path.replace('_test', '_train').replace('/test/', '/train/')
            train = pd.read_csv(train_path)
            ds.add_default(train)

        rtime = ds.rtime
        nof_rules = ds.count_rules()
        nof_lits = ds.count_lits()
        tot_size = ds.count_tot_size()

        ds.accry_overlap(test)
        acry = ds.acry
        overlap = ds.overlap
        orate = ds.overlap_rate
        explsize = ds.xp_size

        info['stats'][dtname] = {'status': status,
                                 'status2': status2,
                                 'rtime': rtime,
                                 'acry': acry,
                                 'overlap': overlap,
                                 'overlaprate': orate,
                                 'rules': nof_rules,
                                 'lits': nof_lits,
                                 'totsize': tot_size,
                                 'explsize': explsize}

    saved_dir = '../stats/real_status'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    saved_file = '{0}/{1}.json'.format(saved_dir, conf)
    with open(saved_file, 'w') as f:
        json.dump(info, f, indent=4)

    saved_dir = '../stats/true_status'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    saved_file = '{0}/{1}.json'.format(saved_dir, conf)

    dtnames = list(info['stats'].keys())
    for dtname in dtnames:
        info['stats'][dtname]['status'] = True
    with open(saved_file, 'w') as f:
        json.dump(info, f, indent=4)
        pass

def bt_acry(dtnames):

    def traverse(tree, nof_node=0, nof_leaf=0):
        nof_node += 1
        if tree.children:
            nof_node, nof_leaf = traverse(tree.children[0], nof_node, nof_leaf)
            nof_node, nof_leaf = traverse(tree.children[1], nof_node, nof_leaf)
        else:
            nof_leaf += 1

        return nof_node, nof_leaf

    label = 'bt'
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}
    for dtname in dtnames:

        for i in range(1, 6):
            dtname_ = '{0}_train{1}'.format(dtname, i)
            dt = '../datasets/train/{0}/{1}_data.csv'.format(dtname, dtname_)
            test = dt.replace('/train/', '/test/').replace('_train{0}'.format(i), '_test{0}'.format(i))
            if os.path.isfile(dt):
                model = './temp/{0}_data/{0}_data_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl'.format(dtname_)
            else:
                model = './temp/{0}/{0}_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl'.format(dtname_)
                test = test.replace('_data.csv', '.csv')

            log = '../logs/bt/' + model.rsplit('/', maxsplit=1)[-1].replace('.mod.pkl', '.log')
            with open(log, 'r') as f:
                lines = f.readlines()
            lines = list(filter(lambda l: 'rtime: ' in l, lines))
            rtime = float(lines[0].split(':')[-1])

            options = Options(['parse_logs.py', model, model])
            options.encode = 'maxsat'

            xgb = XGBooster(options, from_model=options.files[0])

            # encode it and save the encoding to another
            xgb.encode(test_on=options.explain)

            test_df = pd.read_csv(test)

            X_test = test_df.iloc[ : , :-1]
            X_test = np.expand_dims(X_test, axis=0)[0]
            X_test = xgb.transform(X_test)
            y_test = test_df.iloc[:, -1]
            yhat = xgb.model.predict(X_test)
            assert len(yhat) == len(y_test)
            test_accuracy = [1 if yhat[row] == y_test[row] else 0 for row in range(y_test.shape[0])]
            test_accuracy = round(statistics.mean(test_accuracy) * 100, 2)

            ensemble = TreeEnsemble(xgb.model,
                                    xgb.extended_feature_names_as_array_strings,
                                    nb_classes=xgb.num_class)

            nof_nodes = []
            nof_leafs = []
            # traversing and encoding each tree
            for i, tree in enumerate(ensemble.trees):
                nof_node, nof_leaf = traverse(tree)
                nof_nodes.append(nof_node)
                nof_leafs.append(nof_leaf)

            info['stats'][dtname_] = {'status': True,
                                      'status2': True,
                                      'acry': test_accuracy,
                                      'rtime': rtime,
                                      'rules': sum(nof_leafs),
                                      'lits': sum(nof_nodes) - sum(nof_leafs),
                                      'totsize': sum(nof_nodes)}

    saved_dir = '../stats/real_status'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    saved_file = '{0}/{1}.json'.format(saved_dir, 'bt')
    with open(saved_file, 'w') as f:
        json.dump(info, f, indent=4)

    saved_dir = '../stats/true_status'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    saved_file = '{0}/{1}.json'.format(saved_dir, 'bt')
    with open(saved_file, 'w') as f:
        json.dump(info, f, indent=4)

def cactus(files, keys, lloc = '"upper left"', lncol=1, solved=False, ylog=False, xmax=None,
           shape='long', ymin=0, ymax=None, font_sz=9, width=1.0, ms=5, font='cmr', status='status',
           avg=False, show=False, exclude={}, sort=True, bt=False):
    for key in keys:
        saved_dir = '../plots'
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        pdffn = '{0}/{1}{2}{3}.pdf'.format(saved_dir, key, '_solved' if solved else '',
                                           '_avg' if avg else '')

        if solved:
            # detect dataset solvable by all competitors
            for i, fl in enumerate(files):
                with open(fl, 'r') as f:
                    info = json.load(f)
                s = info['stats'].keys()
                if i == 0:
                    solved_datasets = set(s)
                s = set(filter(lambda l: info['stats'][l][status], s))
                solved_datasets = solved_datasets.intersection(s)

            solved_datasets = solved_datasets.difference(exclude)

            # save solvable information to temp files
            files_ = list(map(lambda l: str(l) + '.json', range(len(files))))

            if show:
                print(' & '.join(sorted(solved_datasets)))

            solved_stats = {}
            for i, fl in enumerate(files_):
                with open(files[i], 'r') as f:
                    info = json.load(f)
                info['stats'] = {ss: info['stats'][ss] for ss in sorted(solved_datasets) if ss not in exclude}

                solved_stats[info['preamble']['prog_alias']] = info['stats']
                if show:
                    print(' {0} & {1}'.format(info['preamble']['prog_alias'],
                                              ' & '.join([str(round(info['stats'][ss][key], 2)) for ss in sorted(solved_datasets)])))
                with open(str(i) + '.json', 'w') as f:
                    json.dump(info, f, indent=4)

        else:
            files_ = files

        xlabel = 'Datasets'
        if not xmax:
            xmax = 70 if key == 'acry' and solved else None
        maxkey = cmptmax(files_, key)
        if key == 'acry':
            maxkey = 110


        # "upper left"
        #--font-sz 12
        if key == 'acry':
            key_ = '"Accuray (\%)"'
        elif key == 'rtime':
            key_ = '"CPU time (s)"'
        elif key == 'totsize':
            key_ = '"Total number of literals"'
        elif key == 'explsize':
            key_ = '"Explanation size"'
        else:
            key_ = key

        cmd = f'python ./gnrt_plots/mkplot/mkplot.py -l --font-sz {font_sz} ' \
              f'--shape {shape} --lloc {lloc} --lncol {lncol} --legend prog_alias ' \
              f'--xlabel {xlabel} -b pdf --save-to {pdffn} -k {key} --ylabel {key_} ' \
              f'--width {width} --ms {ms}'
        if bt:
            cmd += ' --bt'
        if sort:
            cmd += ' --sort'
        if font:
            cmd += f' --font "{font}"'
        if ylog:
            cmd += ' --ylog'
        if xmax:
            cmd += f' --xmax {xmax}'
        if ymax:
            cmd += f' --ymax {ymax}'
        if key == 'acry':
            cmd += ' --reverse'
        if maxkey:
            cmd += f' -t {maxkey}'
        if ymin > 0:
            cmd += ' --ymin {0} '.format(ymin)

        cmd = cmd + ' ' + ' '.join(files_)
        print('\ncmd:', cmd)
        os.system(cmd)
        #print()

def scatter(files, keys, solved=False, xlog=False, xmin=0, font_sz = 13,
            shape='squared', alpha=0.1, tol_loc=None, ymax=None, timeout=None,
            ymin=0, tlabel=None, scattersz=None, status='status', font='cmr',
            avg=False, exclude={}):
    for key in keys:

        saved_dir = '../plots'
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        file0_ = files[0].rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0]
        file0_ = file0_.replace('btcmpltrainsortupdate', 'cpl')
        file0_ = file0_.replace('ra', 'l')
        file0_ = file0_.replace('rrulewght', 'r')
        file0_ = file0_.replace('rrule', 'r')

        pdffn = '{0}/{1}_{2}_{3}{4}.pdf'.format(saved_dir, key,
                                             file0_,
                                             files[1].rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0],
                                                '_avg' if avg else '')
        maxkey = cmptmax(files, key)
        maxkey = maxkey if maxkey >= 1 else 1

        if solved:
            for i, fl in enumerate(files):
                with open(fl, 'r') as f:
                    info = json.load(f)
                s = info['stats'].keys()
                if i == 0:
                    solved_datasets = set(s)
                #s = set(filter(lambda l: info['stats'][l]['status2'], s))
                s = set(filter(lambda l: info['stats'][l][status], s))
                solved_datasets = solved_datasets.intersection(s)

            solved_datasets = solved_datasets.difference(exclude)

            files_ = list(map(lambda l: str(l) + '.json', range(len(files))))
            for i, fl in enumerate(files_):
                with open(files[i], 'r') as f:
                    info = json.load(f)

                info['stats'] = {ss: info['stats'][ss] for ss in solved_datasets}
                for ss in solved_datasets:
                    #info['stats'][ss]['status'] = info['stats'][ss]['status2']
                    info['stats'][ss]['status'] = info['stats'][ss][status]
                with open(str(i) + '.json', 'w') as f:
                    json.dump(info, f, indent=4)
        else:
            files_ = files

        cmd = 'python ./gnrt_plots/mkplot/mkplot.py -l -p scatter -b pdf ' \
              f'--tol-loc before --save-to {pdffn} -k {key} --font-sz {font_sz} ' \
              f'--shape {shape} -a {alpha} -t {maxkey}'
        if font:
            cmd += f' --font "{font}"'
        if xlog:
            cmd += ' --xlog --ylog '
        if xmin > 0:
            cmd += ' --xmin {0} --ymin {0} '.format(xmin)
        if tol_loc:
            cmd += f' --tol-loc {tol_loc}'
        else:
            cmd += f' --tol-loc none'
        if ymax:
            cmd += f' --ymax {ymax}'
        if timeout:
            cmd += f' --timeout {timeout}'
        if tlabel:
            cmd += f' --tlabel {tlabel}'
        if scattersz:
            cmd += f' --scattersz {scattersz}'

        cmd = cmd + ' ' + ' '.join(files_)
        print('\ncmd:\n{0}'.format(cmd))
        os.system(cmd)
        #print()

def get_max(files):
    maxstats = collections.defaultdict(lambda :
                                       collections.defaultdict(lambda : []))
    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)

        for train in info['stats']:
            train_stats = info['stats'][train]
            dt = train.rsplit('_train', maxsplit=1)[0]

            for k, v in train_stats.items():
                maxstats[dt][k].append(v)

    for dt in maxstats:
        for k in maxstats[dt]:
            maxstats[dt][k] = {'max': max(maxstats[dt][k]),
                               'min': min(maxstats[dt][k])}
    return maxstats

def get_avg(file, maxstats):
    with open(file, 'r') as f:
        info = json.load(f)
    new_info = {}
    new_info['preamble'] = info['preamble']
    new_info['stats'] = collections.defaultdict(lambda : {})
    dt2stats = collections.defaultdict(lambda :
                                       collections.defaultdict(lambda : []))

    for train in info['stats']:
        train_stats = info['stats'][train]
        dt = train.rsplit('_train', maxsplit=1)[0]

        for k, v in train_stats.items():
            dt2stats[dt][k].append(v)

    dts = sorted(dt2stats.keys())

    for dt in dts:
        status = sum(dt2stats[dt]['status'])
        status2 = sum(dt2stats[dt]['status2'])
        if status in (0, 5):
            new_info['stats'][dt]['status'] = dt2stats[dt]['status'][0]
            new_info['stats'][dt]['status2'] = dt2stats[dt]['status2'][0]
        else:
            new_info['stats'][dt]['status'] = True
            new_info['stats'][dt]['status2'] = True

        other_keys = set(dt2stats[dt].keys()).difference({'status', 'status2'})
        for k in other_keys:
            if status in (0, 5):
                v = dt2stats[dt][k]
            else:
                v = dt2stats[dt][k]
                if k == 'overlap':
                    if status == len(v):
                        v_ = []
                        for sid, s in enumerate(dt2stats[dt]['status']):
                            if s:
                                v_.append(v.pop(0))
                            else:
                                v_.append(0)
                        v = v_

                assert len(v) == 5

                v_ = []
                #print('{0}: {1}; max: {2}'.format(k, v, maxstats[dt][k]))
                for sid, s in enumerate(dt2stats[dt]['status']):
                    if s:
                        v_.append(v[sid])
                    #if not s:
                    #    if k in ('totsize', 'overlaprate', 'explsize',
                    #             'lits', 'rules', 'overlap'):
                    #        # max
                    #        v[sid] = maxstats[dt][k]['max']
                    #    #elif k in ('acry'):
                    #    #    # min
                    #    #    v[sid] = maxstats[dt][k]['min']
                    #    #    pass
                    #    else:
                    #        assert k in ('rtime', 'acry')

                #print('{0}: {1}; max: {2}\n'.format(k, v, maxstats[dt][k]))
                v = v_
                #print('{0}: {1}; max: {2}\n'.format(k, v, maxstats[dt][k]))
                #print('avg:', statistics.mean(v))
            avg = statistics.mean(v)
            new_info['stats'][dt][k] = avg

    saved_dir = file.rsplit('/', maxsplit=1)[0] + '/avg'

    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    saved_file = saved_dir + '/' + file.rsplit('/', maxsplit=1)[-1]

    with open(saved_file, 'w') as f:
        json.dump(new_info, f, indent=4)

def table_row(files, datasets, key):
    print(key)
    for file in files:
        if 'bt.json' in file and key != 'acry':
            continue
        with open(file, 'r') as f:
            info = json.load(f)
        appr = info['preamble']['prog_alias']
        row = [appr]
        for dt in datasets:
            row.append(str(info['stats'][dt][key]))
        print(' & '.join(row))
    print()

if __name__ == '__main__':
    default = True

    """
    
    Parse Decision set logs
    
    """
    logs = collections.defaultdict(lambda: [])

    for root, dirs, files in os.walk('../logs'):
        for file in files:
            if file.endswith('log'):
                conf = root.rsplit('/')[-1]

                dt = file.split('_train', maxsplit=1)[0] + '_train'
                logs[conf].append(os.path.join(root, file))
    confs = list(logs.keys())

    for conf in confs:
        if 'sparse' in conf:
            for log in logs[conf]:
                conf_ = 'sp'
                if '_0.005' in log:
                    conf_ += '0.005'
                    if '_approx' in log:
                        conf_ += 'ceil'
                elif '_0.05' in log:
                    conf_ += '0.05'
                    if '_approx' in log:
                        conf_ += 'ceil'
                elif '_0.5' in log:
                    conf_ += '0.5'
                    if '_approx' in log:
                        conf_ += 'ceil'

                logs[conf_].append(log)
            del logs[conf]
        elif 'imli' in conf:
            for log in logs[conf]:
                conf_ = 'imli'
                if '_1.log' in log:
                    conf_ += '1'
                elif '_16.log' in log:
                    conf_ += '16'
                logs[conf_].append(log)
            del logs[conf]
        elif 'compile' in conf:
            for log in logs[conf]:
                conf_ = 'cmpl'
                if '_bg_' in log:
                    bg = log.rsplit('_bg_', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].split('size')[-1].split('_')[0]
                    conf_ += 'size{0}'.format(bg)
                else:
                    bg = None

                if '_clocal' in log:
                    conf_ += 'train'

                if '_fsort' in log:
                    conf_ += 'sort'

                if '_fqupdate' in log:
                    conf_ += 'update'

                if '_rrule' in log:
                    conf_ += 'rrule'
                    if '_wght' in log:
                        conf_ += 'wght'

                if 'reduce_before' in log:
                    conf_ += 'rb'
                elif 'reduce_after' in log:
                    conf_ += 'ra'

                if '_0.005' in log:
                    conf_ += '0.005'
                    if '_approx' in log:
                        conf_ += 'ceil'
                elif '_0.05' in log:
                    conf_ += '0.05'
                    if '_approx' in log:
                        conf_ += 'ceil'
                elif '_0.5' in log:
                    conf_ += '0.5'
                    if '_approx' in log:
                        conf_ += 'ceil'

                logs[conf_].append(log)
            del logs[conf]

    confs = list(logs.keys())

    """
    Parse DS logs
    """
    for conf in confs:
        if conf == 'bt':
            continue
        print(conf)
        files = logs[conf]
        parse_dslog(files, conf, default)
        pass

    """
    bt acry
    """
    dtnames = set()
    for root, dirs, files in os.walk('../datasets/train'):
        for file in files:
            if file.endswith('.csv'):
                dtname = file.split('_train', maxsplit=1)[0]
                dt = dtname + '_train'
                dtnames.add(dtname)
    bt_acry(dtnames)

    """
    bt + cmpl runtime
    """
    files = glob.glob('../stats/real_status/cmpl*.json')
    files = list(files) + list(glob.glob('../stats/true_status/cmpl*.json'))
    bt_file = '../stats/real_status/bt.json'
    with open(bt_file, 'r') as f:
        bt_info = json.load(f)

    for file in files:
        with open(file, 'r') as f:
            ds_info = json.load(f)

        for dt in ds_info['stats']:
            ds_info['stats'][dt]['rtime'] += bt_info['stats'][dt]['rtime']

        new_file = file.rsplit('/')
        new_file[-1] = 'bt' + new_file[-1]
        new_file = '/'.join(new_file)

        with open(new_file, 'w') as f:
            json.dump(ds_info, f, indent=4)

    # accuracy
    # use all true file where accuracy is 0 if datasets unsolved
    saved_dir = '../stats/true_status'
    acry_opt = saved_dir + '/opt.json'
    acry_twostg = saved_dir + '/twostg.json'
    acry_sparse = glob.glob(saved_dir + '/sp*.json')
    acry_cmpl = sorted(glob.glob(saved_dir + '/btcmpltrain*.json'), reverse=True)
    acry_bt = saved_dir + '/bt.json'
    acry_cn2 = saved_dir + '/cn2.json'
    acry_ripper = saved_dir + '/ripper.json'
    acry_imli = sorted(glob.glob(saved_dir + '/imli*.json'), reverse=True)
    acry_ids = saved_dir + '/ids.json'
    acry_others = [acry_cn2, acry_ripper] + acry_imli + [acry_twostg, acry_ids] + acry_sparse + [acry_opt]
    acry_files = [acry_bt] + acry_cmpl + acry_others
    cactus(acry_files, keys=['acry'], lloc='"lower right"', lncol=1, ylog=False, font_sz=15,
           ms=7, width=1.5, xmax=450, shape='squared', sort=False, bt=True)

    # rtime
    # all true files
    saved_dir = '../stats/real_status'
    rtime_opt = saved_dir + '/opt.json'
    rtime_twostg = saved_dir + '/twostg.json'
    rtime_sparse = glob.glob(saved_dir + '/sp*.json')
    rtime_cmpl = sorted(glob.glob(saved_dir + '/btcmpltrain*.json'), reverse=True)
    rtime_cn2 = saved_dir + '/cn2.json'
    rtime_ripper = saved_dir + '/ripper.json'
    rtime_imli = sorted(glob.glob(saved_dir + '/imli*.json'), reverse=True)
    rtime_ids = saved_dir + '/ids.json'
    rtime_others = [rtime_cn2, rtime_ripper] + rtime_imli + [rtime_twostg, rtime_ids] + rtime_sparse + [rtime_opt]
    rtime_files = rtime_cmpl + rtime_others
    keys = ['rtime']
    cactus(rtime_files, keys=keys, lloc='"lower right"', font_sz=15, ms=7,
           width=1.5, xmax=470, ymax=3600, shape='squared', sort=False)

    # size
    saved_dir = '../stats/real_status'
    size_opt = saved_dir + '/opt.json'
    size_twostg = saved_dir + '/twostg.json'
    size_sparse = glob.glob(saved_dir + '/sp*.json')
    size_cmpl = glob.glob(saved_dir + '/btcmpltrain*.json')
    size_bt = saved_dir + '/bt.json'
    size_cn2 = saved_dir + '/cn2.json'
    size_ripper = saved_dir + '/ripper.json'
    size_imli = sorted(glob.glob(saved_dir + '/imli*.json'), reverse=True)
    size_ids = saved_dir + '/ids.json'
    size_others = [size_cn2, size_ripper] + size_imli + [size_twostg, size_ids] + size_sparse + [size_opt]
    size_files = size_cmpl + size_others

    cactus(size_files[:-1], keys=['totsize', 'explsize'], ylog=True,
           ymin=1, xmax=450, lloc='"lower right"', shape='squared', font_sz=15,
           ms=7, width=1.5, sort=False)

    # expl size scatter plot
    keys = ['explsize']
    for cmpl_file in size_cmpl:
        if 'cmpltrainsortupdaterrulewghtra.json' not in cmpl_file:
            continue
        for file in size_others:
            if 'opt' in file or '/sp' in file or 'imli1.json' in file:
                continue
            scatter([cmpl_file, file], keys=keys, xmin=1, xlog=True, solved=True,
                    scattersz=130, font_sz=26, alpha=0.3)
            pass

    # acry scatter plot
    saved_dir = '../stats/true_status'
    acry_opt = saved_dir + '/opt.json'
    acry_twostg = saved_dir + '/twostg.json'
    acry_sparse = glob.glob(saved_dir + '/sp*.json')
    acry_cmpl = glob.glob(saved_dir + '/btcmpl*.json')
    acry_bt = saved_dir + '/bt.json'
    acry_cn2 = saved_dir + '/cn2.json'
    acry_ripper = saved_dir + '/ripper.json'
    acry_imli = glob.glob(saved_dir + '/imli*.json')
    acry_others = [acry_opt, acry_twostg, acry_cn2, acry_ripper] + acry_sparse + acry_imli
    acry_files = acry_others + acry_cmpl + [acry_bt]
    for cmpl_file in acry_cmpl:
        if 'cmpltrainsortupdaterrulewghtra.json' not in cmpl_file:
            continue
        for file in acry_others:
            if 'opt' in file or '/sp' in file or 'imli1.json' in file:
                continue
            scatter([cmpl_file, file], keys=['acry'], xmin=0,
                    scattersz=110, font_sz=26, alpha=0.3)
            pass

    """
    Exhaustive compilation plots
    """
    # pre accuracy
    # use all true file where accuracy is 0 if datasets unsolved
    saved_dir = '../stats/real_status'

    acry_opt = saved_dir + '/opt.json'
    acry_twostg = saved_dir + '/twostg.json'
    acry_sparse = glob.glob(saved_dir + '/sp*.json')
    acry_cmpl = [saved_dir + '/btcmpl.json']
    acry_bt = saved_dir + '/bt.json'
    acry_cn2 = saved_dir + '/cn2.json'
    acry_ripper = saved_dir + '/ripper.json'
    acry_imli = glob.glob(saved_dir + '/imli*.json')
    acry_ids = saved_dir + '/ids.json'
    acry_others = [acry_cn2, acry_ripper] + acry_imli + [acry_twostg, acry_ids] + acry_sparse + [acry_opt]
    acry_files = [acry_bt] + acry_cmpl + acry_others
    cactus(acry_files, keys=['acry'], lloc='"lower right"', lncol=1, ylog=False, font_sz=18,
           ms=9, width=1.8, xmax=50, shape='squared', solved=True, status='status2', sort=False, bt=True)

    scatter(acry_cmpl + [acry_bt], keys=['acry'], xmin=1, solved=True,  # xlog=True,
            scattersz=130, font_sz=18, alpha=0.3, status='status2')

    # pre size
    saved_dir = '../stats/real_status'
    size_opt = saved_dir + '/opt.json'
    size_twostg = saved_dir + '/twostg.json'
    size_sparse = glob.glob(saved_dir + '/sp*.json')
    size_cmpl = [saved_dir + '/btcmpl.json']
    size_bt = saved_dir + '/bt.json'
    size_cn2 = saved_dir + '/cn2.json'
    size_ripper = saved_dir + '/ripper.json'
    size_imli = glob.glob(saved_dir + '/imli*.json')
    size_ids = saved_dir + '/ids.json'
    size_others = [size_cn2, size_ripper] + size_imli + [size_twostg, size_ids] + size_sparse + [size_opt]
    size_files = size_cmpl + size_others

    cactus(size_files, keys=['totsize', 'explsize'], ylog=True,
          ymin=1, xmax=50, lloc='"lower right"', shape='squared', font_sz=18,
          ms=9, width=1.8, solved=True, status='status2', sort=False)

    """
    convert pair to average
    """

    files = []
    files.extend(list(glob.glob('../stats/real_status/*.json')))
    files.extend(list(glob.glob('../stats/true_status/*.json')))
    files.sort()

    maxstats = get_max(files)
    for file in files:
        get_avg(file, maxstats)

    # table stats
    datasets = ['cardiotocography', 'hayes-roth',  'iris', 'new-thyroid', 'orbit', 'zoo']

    saved_dir = '../stats/true_status/avg/'
    acry_opt = saved_dir + '/opt.json'
    acry_twostg = saved_dir + '/twostg.json'
    acry_sparse = glob.glob(saved_dir + '/sp*.json')
    acry_cmpl = glob.glob(saved_dir + '/btcmpltrain*.json')
    acry_bt = saved_dir + '/bt.json'
    acry_cn2 = saved_dir + '/cn2.json'
    acry_ripper = saved_dir + '/ripper.json'
    acry_imli = glob.glob(saved_dir + '/imli*.json')
    acry_ids = saved_dir + '/ids.json'
    acry_others = [acry_opt, acry_twostg, acry_cn2, acry_ripper, acry_ids] + acry_sparse + acry_imli
    acry_files = acry_others + acry_cmpl + [acry_bt]

    for key in ['acry', 'totsize', 'explsize']:
        table_row(acry_files, datasets, key)

    """
    average plots
    """
    # accuracy
    # use all true file where accuracy is 0 if datasets unsolved
    saved_dir = '../stats/true_status/avg/'
    acry_opt = saved_dir + '/opt.json'
    acry_twostg = saved_dir + '/twostg.json'
    acry_sparse = glob.glob(saved_dir + '/sp*.json')
    acry_cmpl = sorted(glob.glob(saved_dir + '/btcmpltrain*.json'), reverse=True)
    acry_bt = saved_dir + '/bt.json'
    acry_cn2 = saved_dir + '/cn2.json'
    acry_ripper = saved_dir + '/ripper.json'
    acry_imli = sorted(glob.glob(saved_dir + '/imli*.json'), reverse=True)
    acry_ids = saved_dir + '/ids.json'
    acry_others = [acry_cn2, acry_ripper] + acry_imli + [acry_twostg, acry_ids] + acry_sparse + [acry_opt]
    acry_files = [acry_bt] + acry_cmpl + acry_others
    cactus(acry_files, keys=['acry'], lloc='"lower right"', lncol=1, ylog=False, font_sz=15,
           ms=7, width=1.5, xmax=90, shape='squared', avg=True, sort=False, bt=True)

    # rtime
    # all true files
    saved_dir = '../stats/real_status/avg'
    rtime_opt = saved_dir + '/opt.json'
    rtime_twostg = saved_dir + '/twostg.json'
    rtime_sparse = glob.glob(saved_dir + '/sp*.json')
    rtime_cmpl = sorted(glob.glob(saved_dir + '/btcmpltrain*.json'), reverse=True)
    rtime_cn2 = saved_dir + '/cn2.json'
    rtime_ripper = saved_dir + '/ripper.json'
    rtime_imli = sorted(glob.glob(saved_dir + '/imli*.json'), reverse=True)
    rtime_ids = saved_dir + '/ids.json'
    rtime_others = [rtime_cn2, rtime_ripper] + rtime_imli + [rtime_twostg, rtime_ids] + rtime_sparse + [rtime_opt]
    rtime_files = rtime_cmpl + rtime_others

    keys = ['rtime']
    cactus(rtime_files, keys=keys, lloc='"lower right"', font_sz=15, ms=7,
          width=1.5, xmax=90, ymax=3600, shape='squared', avg=True, sort=True)

    # size
    saved_dir = '../stats/real_status/avg'
    size_opt = saved_dir + '/opt.json'
    size_twostg = saved_dir + '/twostg.json'
    size_sparse = glob.glob(saved_dir + '/sp*.json')
    size_cmpl = sorted(glob.glob(saved_dir + '/btcmpltrain*.json'), reverse=True)
    size_bt = saved_dir + '/bt.json'
    size_cn2 = saved_dir + '/cn2.json'
    size_ripper = saved_dir + '/ripper.json'
    size_imli = sorted(glob.glob(saved_dir + '/imli*.json'), reverse=True)
    size_ids = saved_dir + '/ids.json'
    size_others = [size_cn2, size_ripper] + size_imli + [size_twostg, size_ids] + size_sparse + [size_opt]
    size_files = size_cmpl + size_others

    cactus(size_files, keys=['totsize', 'explsize'], ylog=True,
           ymin=1, xmax=90, lloc='"lower right"', shape='squared', font_sz=15,
           ms=7, width=1.5, avg=True, sort=False)

    keys = ['explsize']
    for cmpl_file in size_cmpl:
       if 'cmpltrainsortupdaterrulewghtra.json' not in cmpl_file:
           continue
       for file in size_others:
           if 'opt' in file or '/sp' in file or 'imli1.json' in file:
               continue
           scatter([cmpl_file, file], keys=keys, xmin=1, xlog=True, solved=True,
                   scattersz=250, font_sz=26, alpha=0.3, avg=True)
           pass

    saved_dir = '../stats/true_status/avg'
    acry_opt = saved_dir + '/opt.json'
    acry_twostg = saved_dir + '/twostg.json'
    acry_sparse = glob.glob(saved_dir + '/sp*.json')
    acry_cmpl = glob.glob(saved_dir + '/btcmpl*.json')
    acry_bt = saved_dir + '/bt.json'
    acry_cn2 = saved_dir + '/cn2.json'
    acry_ripper = saved_dir + '/ripper.json'
    acry_imli = glob.glob(saved_dir + '/imli*.json')
    acry_ids = saved_dir + '/ids.json'
    acry_others = [acry_opt, acry_twostg, acry_cn2, acry_ripper, acry_ids] + acry_sparse + acry_imli
    acry_files = acry_others + acry_cmpl + [acry_bt]
    for cmpl_file in acry_cmpl:
       if 'cmpltrainsortupdaterrulewghtra.json' not in cmpl_file:
           continue
       for file in acry_others:
           if 'opt' in file or '/sp' in file or 'imli1.json' in file:
               continue
           scatter([cmpl_file, file], keys=['acry'], xmin=0,
                   scattersz=230, font_sz=26, alpha=0.3, avg=True)
           pass

    """
    Exhaustive compilation plots
    """
    #pre accuracy
    #use all true file where accuracy is 0 if datasets unsolved
    saved_dir = '../stats/real_status/avg'
    acry_opt = saved_dir + '/opt.json'
    acry_twostg = saved_dir + '/twostg.json'
    acry_sparse = glob.glob(saved_dir + '/sp*.json')
    acry_cmpl = [saved_dir + '/btcmpl.json']
    acry_bt = saved_dir + '/bt.json'
    acry_cn2 = saved_dir + '/cn2.json'
    acry_ripper = saved_dir + '/ripper.json'
    acry_imli = glob.glob(saved_dir + '/imli*.json')
    acry_ids = saved_dir + '/ids.json'
    acry_others = [acry_opt, acry_twostg, acry_cn2, acry_ripper, acry_ids] + acry_sparse + acry_imli
    acry_files = acry_others + acry_cmpl + [acry_bt]
    cactus(acry_files, keys=['acry'], lloc='"lower right"', lncol=1, ylog=False, font_sz=18,
          ms=9, width=1.8, xmax=20, shape='squared', solved=True, status='status2', avg=True, show=True,
           exclude={'post-operative'})
    scatter(acry_cmpl + [acry_bt], keys=['acry'], xmin=1, solved=True,  # xlog=True,
           scattersz=130, font_sz=26, alpha=0.3, status='status2', avg=True)

    #pre size
    saved_dir = '../stats/real_status/avg'
    size_opt = saved_dir + '/opt.json'
    size_twostg = saved_dir + '/twostg.json'
    size_sparse = glob.glob(saved_dir + '/sp*.json')
    size_cmpl = [saved_dir + '/btcmpl.json']
    size_bt = saved_dir + '/bt.json'
    size_cn2 = saved_dir + '/cn2.json'
    size_ripper = saved_dir + '/ripper.json'
    size_imli = glob.glob(saved_dir + '/imli*.json')
    size_ids = saved_dir + '/ids.json'
    size_others = [size_opt, size_twostg, size_cn2, size_ripper, size_ids] + size_sparse + size_imli
    size_files = size_others + size_cmpl + [size_bt]

    cactus(size_files[:-1], keys=['totsize', 'explsize'], ylog=True,
         ymin=1, xmax=20, lloc='"lower right"', shape='squared', font_sz=18,
         ms=9, width=1.8, solved=True, status='status2', avg=True,
           show=True, exclude={'post-operative'})
    exit()
