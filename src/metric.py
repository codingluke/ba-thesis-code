# coding: utf-8

import os, sys, json
import numpy as np
import pdb
from timeit import default_timer as timer
from collections import OrderedDict
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import re
from pandas.tools.plotting import table
plt.style.use('ggplot')

class MetricReader(object):

    def __init__(self, config_dir_path=None):
        self.options = get_options(config_dir_path)
        self.connection, self.db, self.metrics, self.meta, \
            self.trainings = self.__connect()

    def get_records(self, job_id=None, typ='train'):
        type = re.compile(r'^%s' % typ, re.I)
        cursor = self.metrics.find({'job_id' : job_id,
                                    'type' : { '$regex' : type}})
        return pd.DataFrame(list(cursor))


    def get_best_records(self):
        None

    def get_best_job_id(self):
        None

    def plot_pretrain(self, job_id=None):
        df = self.get_records(job_id=job_id, typ='pretrain')
        fig, ax = plt.subplots(1, 1)
        num_trainings = len(df[df['epoch'] == 0])
        print num_trainings

        for i in xrange(num_trainings):
            training = df[df['type'] == 'pretrain_%d' % i]
            training[['cost', 'epoch']].plot(x='epoch', y='cost',
                                   ax=ax)
        ax.set_title('Autoencoder Vortraining')
        ax.set_ylabel('Cross-Entropy')
        ax.set_xlabel('Epochen')
        # ymin, ymax = ax.get_ylim()
        # epochs = [df[df['epoch']==e]['iteration'].values[-1]
                  # for e in xrange(df['epoch'].max())]
        # ax.vlines(x=epochs, ymin=[ymin], ymax=[ymax], label='epochs', linestyle='dotted')
        # min = df['validation_accuracy'].min()
        # min_val = df[df['validation_accuracy']==min]
        # min_x = min_val['iteration'].values[0]
        # min_y = min_val['validation_accuracy'].values[0]
        # xmin, xmax = ax.get_xlim()
        # ax.annotate('min at (%f)' % min_y, xy=(min_x, min_y), xytext=(xmax/2, ymax/2),
                    # arrowprops=dict(facecolor='black', shrink=0.05))
        # ax.plot(min_x, min_y, 'o', color="k")
        l = ax.legend()
        meta = self.get_job_metadata(job_id=job_id)
        layers = meta['layers'].values[0].split('-')
        for i in xrange(num_trainings):
            l.get_texts()[i].set_text(u"Layer %d : %s" % (i,layers[i]) )
        plt.show()

    def get_job_metadata(self, job_id=None):
        cursor = self.trainings.find({'job_id' : job_id})
        return pd.DataFrame(list(cursor))

    def plot_epoches(self, job_id=None, first=False, title=None):
        index = 0 if first else -1
        if not title: title = "Lernphase Job %d" % job_id

        df = self.get_records(job_id=job_id)
        fig, ax = plt.subplots(1, 1)
        epochs = [df[df['epoch']==e]['iteration'].values[index]
                  for e in xrange(df['epoch'].max())]
        df = df[df['iteration'].isin(epochs)].reset_index()
        df[['cost', 'validation_accuracy']].plot(
                y=['cost', 'validation_accuracy'],
                title=title,
                ax=ax,
                subplots=True)
        seconds_epoche = int(df[df['iteration']==epochs[0]]['second'].values[0])

        ymin, ymax = ax.get_ylim()
        min = df['validation_accuracy'].min()
        min_val = df[df['validation_accuracy']==min]
        min_x = min_val.index.values[0]
        min_y = min_val['validation_accuracy'].values[0]
        xmin, xmax = ax.get_xlim()
        ax.annotate('Minimum (%f)' % min_y, xy=(min_x, min_y), xytext=(xmax/2, ymax/2),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        ax.plot(min_x, min_y, 'o', color="k")
        ax.set_xlabel('Epochen')
        # l = ax.legend()
        # l.get_texts()[0].set_text(u"Trainings Kosten (cross-entropy)")
        # l.get_texts()[1].set_text(u"Validation Präzision (rmse)")
        zeit = '%ds / Epoche' % seconds_epoche
        tb = ax.table(cellText=[['RMSProp'], [0.01], [0.0], ['-'] ,[2], [1000],
                           [500000], [3000000], [500000], [zeit]],
                 rowLabels=['Algorithmus', 'Lernrate', 'L2', 'Dropout',
                            'Rahmen', 'Minibatch', 'Batch', 'Training',
                            'Validation', 'Zeit'],
                 colWidths=[0.2, 0.4], cellLoc='left',
                 loc=None, bbox=[1.25, 0.0, 0.3, 1.0])
        tb.scale(1.5, 1.5)
        plt.show()


    def plot(self, job_id=None):
        df = self.get_records(job_id=job_id)
        fig, ax = plt.subplots(1, 1)
        df[['cost', 'validation_accuracy', 'iteration']].plot(x='iteration',
                                                              y=['cost', 'validation_accuracy'],
                                                              title="Lernphase Job %d" % job_id,
                                                              ax=ax,
                                                              subplots=True)
        ymin, ymax = ax.get_ylim()
        epochs = [df[df['epoch']==e]['iteration'].values[-1]
                  for e in xrange(df['epoch'].max())]
        ax.vlines(x=epochs, ymin=[ymin], ymax=[ymax], label='epochs', linestyle='dotted')
        min = df['validation_accuracy'].min()
        min_val = df[df['validation_accuracy']==min]
        min_x = min_val['iteration'].values[0]
        min_y = min_val['validation_accuracy'].values[0]
        xmin, xmax = ax.get_xlim()
        ax.annotate('min at (%f)' % min_y, xy=(min_x, min_y), xytext=(xmax/2, ymax/2),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        ax.plot(min_x, min_y, 'o', color="k")
        ax.legend()
        plt.show()

    def get_metadata(self):
        experiment_name = self.options['experiment-name']
        return self.meta.find_one({ 'experiment_name' : experiment_name})

    def __connect(self):
        db_name = self.options['database']['name']
        db_address = self.options['database']['address']
        experiment_name = self.options['experiment-name']
        connection = MongoClient(db_address)
        db = connection[db_name]
        metrics = db.db[experiment_name]['metrics']
        meta = db.db['meta']
        trainings = db.db[experiment_name]['trainings']
        # trainings = db.db['trainings']
        return connection, db, metrics, meta, trainings

class MetricRecorder(object):

    def __init__(self, job_id=None, config_dir_path=None):
        self.options = get_options(config_dir_path)
        self.connection, self.db, self.metrics, self.meta, \
            self.trainings = self.__connect()
        if not job_id: job_id = self.get_unique_job_id()
        self.job_id = job_id
        self.experiment_name = self.options['experiment-name']
        self.starttime = timer()
        self.endtime = 0

    def get_unique_job_id(self):
        ids = self.trainings.distinct('job_id')
        if len(ids) > 0:
          return max(self.trainings.distinct('job_id')) + 1
        else:
          return 1


    def start(self):
        self.starttime = timer()

    def stop(self):
        self.endtime = timer()
        return self.duration()

    def duration(self):
        if self.endtime != 0:
            return self.endtime - self.starttime
        else:
            return timer() - self.starttime

    def record(self, job_id=None, cost=None, validation_accuracy=None,
            epoch=None, iteration=None, second=None, type='train',
            eta=None):
        if not job_id: job_id = self.job_id
        if not second: second = self.duration()
        if validation_accuracy:
            validation_accuracy = float(validation_accuracy)
        doc = {
            'job_id' : job_id,
            'cost' : float(cost),
            'validation_accuracy' : validation_accuracy,
            'epoch' : epoch,
            'iteration' : iteration,
            'second' : second,
            'type' : type,
        }
        if eta: doc['eta'] = float(eta)
        self.metrics.insert_one(doc)

    def record_training_info(self, infos=None, job_id=None, second=None):
        if not job_id: job_id = self.job_id
        if not second: second = self.duration()
        infos['job_id'] = job_id
        self.trainings.insert_one(infos)

    def add_experiment_metainfo(self, constants=None, force=None):
        experiment_name = self.options['experiment-name']
        experiment = self.meta.find_one({'experiment_name' :
                                         experiment_name})
        if experiment and force:
            self.meta.remove({'experiment_name' : experiment_name})
        elif experiment:
            return None

        self.meta.insert_one({
            'experiment_name' : experiment_name,
            'constants' : constants})

    def clean_experiment(self):
        self.metrics.drop()
        self.meta.remove({'experiment_name' :
                          self.options['experiment-name'] })

    def clean_job(self, job_id=None):
        if not job_id: job_id = self.job_id
        self.metrics.remove({'job_id' : job_id})

    def __connect(self):
        db_name = self.options['database']['name']
        db_address = self.options['database']['address']
        experiment_name = self.options['experiment-name']
        connection = MongoClient(db_address)
        db = connection[db_name]
        metrics = db.db[experiment_name]['metrics']
        meta = db.db['meta']
        trainings = db.db[experiment_name]['trainings']
        return connection, db, metrics, meta, trainings

def get_options(dir_path):
    # Read in the config file
    expt_dir  = os.path.realpath(os.path.expanduser(dir_path))
    expt_file = ''
    if not os.path.isdir(expt_dir) and not os.path.isfile(expt_dir):
        raise Exception("Cannot find directory or file %s" % expt_dir)
    if os.path.isfile(expt_dir):
        expt_file = expt_dir
    else:
        expt_file = os.path.join(expt_dir, 'config.json')

    try:
        with open(expt_file, 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise Exception("config.json did not load properly. Perhaps a spurious comma?")

    if not os.path.exists(expt_dir):
        sys.stderr.write("Cannot find experiment directory '%s'. "
                         "Aborting.\n" % (expt_dir))
        sys.exit(-1)

    return options
