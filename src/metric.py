# coding: utf-8

import os, sys, json
import numpy as np
import pdb
from timeit import default_timer as timer
from collections import OrderedDict
from pymongo import MongoClient


class MetricReader(object):

    def __init__(self):
        None


class MetricRecorder(object):

    def __init__(self, job_id=None, config_dir_path=None):
        self.job_id = job_id
        self.options = self.__get_options(config_dir_path)
        self.connection, self.db, self.metrics, self.meta = self.__connect()
        self.starttime = timer()
        self.endtime = 0

    def start(self):
        self.starttime = timer()

    def stop(self):
        self.endtime = timer()
        return self.duration()

    def duration(self):
        if self.endtime != 0:
            return self.endtime - self.starttime
        else:
            timer() - self.starttime

    def record(self, job_id=None, cost=None, validation_accuracy=None,
            epoche=None, iteration=None, second=None):
        if not job_id: job_id = self.job_id
        if not second: second = self.duration()
        doc = {
            'job_id' : job_id,
            'cost' : cost,
            'validation_accuracy' : validation_accuracy,
            'epoche' : epoche,
            'iteration' : iteration,
            'second' : second
        }
        self.metrics.insert_one(doc)

    def add_experiment_metainfo(self, constants=None, force=None):
        experiment_name = self.options['experiment-name']
        experiment = self.meta.find_one({'experiment_name' : experiment_name})
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
        return connection, db, metrics, meta

    def __get_options(self, dir_path):
        # Read in the config file
        expt_dir  = os.path.realpath(os.path.expanduser(dir_path))
        if not os.path.isdir(expt_dir):
            raise Exception("Cannot find directory %s" % expt_dir)
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
