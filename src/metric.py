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

    def get_records(self, job_id=None, typ='train', experiment_name=None):
        collection = self.metrics
        if experiment_name:
            collection = self.db.db[experiment_name]['metrics']
        type = re.compile(r'^%s' % typ, re.I)
        cursor = collection.find({'job_id' : job_id,
                                    'type' : { '$regex' : type}})
        return pd.DataFrame(list(cursor))

    def get_best_records(self):
        None

    def get_best_job_id(self):
        None

    def get_experiment_names(self):
        return self.db.collection_names()

    def get_job_metadata(self, job_id=None, experiment_name=None):
        collection = self.trainings
        if experiment_name:
            collection = self.db.db[experiment_name]['trainings']
        cursor = collection.find({'job_id' : job_id})
        return pd.DataFrame(list(cursor))

    def plot_pretrain(self, job_id=None):
        df = self.get_records(job_id=job_id, typ='pretrain')
        fig, ax = plt.subplots(1, 1)
        num_trainings = len(df[df['epoch'] == 0])

        for i in xrange(num_trainings):
            training = df[df['type'] == 'pretrain_%d' % i]
            training[['cost', 'epoch']].plot(x='epoch', y='cost',
                                   ax=ax)
        ax.set_title('Autoencoder Vortraining')
        ax.set_ylabel('Cross-Entropy')
        ax.set_xlabel('Epochen')
        l = ax.legend()
        meta = self.get_job_metadata(job_id=job_id)
        layers = meta['layers'].values[0].split('-')
        for i in xrange(num_trainings):
            l.get_texts()[i].set_text(u"Layer %d : %s" % (i,layers[i]) )
        plt.show()

    def compair_plot(self, job_ids=[], colors=['r', 'g', 'b'],
                     titles=[1, 2, 3], figsize=(9,2.4), xytext=None,
                     experiment_names=[]):
        if experiment_names and len(experiment_names) != len(job_ids):
            raise Exception("all job_ids must have an experiment_name")

        df = pd.DataFrame()
        title = 'Trainingsverlauf Vergleiche'
        ax = None
        cols = []
        for index, job in enumerate(job_ids):
          tmp = self.get_records(job_id=job,
                  experiment_name=experiment_names[index])
          tmp = tmp[['cost', 'validation_accuracy', 'epoch', 'iteration']]
          tmp.columns = ['Trainingskosten-%s' % titles[index],
                        'Validierungskosten-%s' % titles[index],
                        'epoch', 'iteration']
          cols.append('Validierungskosten-%s' % titles[index])
          num_per_epoch = len(tmp[tmp['epoch']==0])
          max_epoche = tmp['epoch'].values.max()
          tmp['Epochen'] = pd.Series(np.linspace(1./num_per_epoch,
                                                 max_epoche, len(tmp)),
                                     index=tmp.index)
          df = df.append(tmp)
          ax = df.plot(
                  x='Epochen',
                  y=['Trainingskosten-%s' % titles[index],
                     'Validierungskosten-%s' % titles[index]],
                  ax=ax, subplots=True, layout=(1,2),
                  figsize=figsize,
                  color=[colors[index], colors[index]])
        plt.subplots_adjust(wspace=0.3, hspace=0.3);

        ax[0].set_title('Trainingskosten')
        ax[0].set_ylabel('Cross-Entropy')
        ax[1].set_title('Validierungskosten')
        ax[1].set_ylabel('RMS-Error')
        l = ax[0].legend()
        l2 = ax[1].legend()
        for i in xrange(len(job_ids)):
            l.get_texts()[i].set_text(titles[i])
            l2.get_texts()[i].set_text(titles[i])

        min = df[cols].min().values.min()
        min_val_col = ''
        for c in cols:
          if len(df[df[c]==min]) > 0: min_val_col = c
        min = df[df[min_val_col]==min]
        min_x = min['Epochen'].values[0]
        min_y = min[min_val_col].values[0]
        xmin, xmax = ax[1].get_xlim()
        ymin, ymax = ax[1].get_ylim()

        if not xytext: xytext = (xmax/2, (ymax+ymin)/2)
        ax[1].annotate('Minimum (%f)' % min_y, xy=(min_x, min_y),
                        xytext=xytext,
                        arrowprops=dict(facecolor='black', shrink=0.05))
        ax[1].plot(min_x, min_y, 'o', color="k")

    def plot(self, job_id=None, experiment_name=None):
        df = self.get_records(job_id=job_id, experiment_name=experiment_name)
        fig, ax = plt.subplots(1, 1)
        m = self.get_job_metadata(job_id=job_id,
                                  experiment_name=experiment_name)
        title = m['layers'].values[0]
        axs = df[['cost', 'validation_accuracy', 'iteration']].plot(
                x='iteration',
                y=['cost', 'validation_accuracy'],
                title= title, ax=ax, subplots=True)
        epochs = [df[df['epoch']==e]['iteration'].values[-1]
                  for e in xrange(df['epoch'].max())]

        ymin, ymax = axs[0].get_ylim()
        axs[0].vlines(x=epochs, ymin=[ymin],
                      ymax=[ymax], label='epochs',
                      linestyle='dotted')
        l = axs[0].legend()
        l.get_texts()[0].set_text(u"Trainings-Kosten")
        l.get_texts()[1].set_text(u"Epochen")
        axs[0].set_ylabel('Cross-Entropy')

        ymin, ymax = axs[1].get_ylim()
        axs[1].vlines(x=epochs, ymin=[ymin], ymax=[ymax],
                label=None, linestyle='dotted')
        l = axs[1].legend()
        l.get_texts()[0].set_text(u"Validierungs-Kosten")
        axs[1].set_ylabel('RMS-Error')
        min = df['validation_accuracy'].min()
        min_val = df[df['validation_accuracy']==min]
        min_x = min_val['iteration'].values[0]
        min_y = min_val['validation_accuracy'].values[0]
        xmin, xmax = axs[1].get_xlim()
        ymin, ymax = axs[1].get_ylim()
        axs[1].annotate('min at (%f)' % min_y, xy=(min_x, min_y),
                        xytext=(xmax/2, (ymax+ymin)/2),
                        arrowprops=dict(facecolor='black', shrink=0.05))
        axs[1].plot(min_x, min_y, 'o', color="k")

        if len(epochs) > 0:
            seconds_epoche = int(df[df['iteration']==epochs[0]]['second'].values[0])
        else: seconds_epoche = 0
        zeit = '%ds / Epoche' % seconds_epoche
        m = self.get_job_metadata(job_id=job_id)
        d = {}
        for col in m.columns: d[col] = m[col].values[0]
        if 'random_mode' in d:
          d['training_data'] = '%d : %s' % (d['training_data'], d['random_mode'])
        if 'eta_min' in d:
            d['eta'] = '%.03f -> %.03f' % (d['eta'], + d['eta_min'])
        tb = axs[1].table(cellText=[[d['algorithm']], [d['eta']],
                                    [d['lmbda']], [d['dropouts']] ,
                                    [d['mini_batch_size']],
                                    [d['training_data']],
                                    [d['validation_data']],
                                    [zeit]],
                 rowLabels=['Algorithmus', 'Lernrate', 'L2', 'Dropout',
                            'Minibatch', 'Training',
                            'Validation', 'Zeit'],
                 colWidths=[0.2, 0.4], cellLoc='left',
                 loc=None, bbox=[1.25, 0.0, 0.3, 2.2])
        tb.scale(1.5, 1.5)
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
