import unittest
import numpy as np
import PIL.Image
import pdb
import os
import pymongo
from timeit import default_timer as timer
from itertools import izip
from collections import OrderedDict

import config
from src.metric import MetricRecorder

@unittest.skipUnless(config.mongodb, 'MongoDB deactivated')
class TestMetricRecorder(unittest.TestCase):

    def setUp(self):
        self.metric = MetricRecorder(
            config_dir_path='./tests/data/spearmint_config/config.json',
            job_id=1)

    def tearDown(self):
        self.metric.clean_experiment()
        ln = len([x for x in self.metric.metrics.find()])
        self.assertEqual(ln, 0)

    def test_load_options(self):
        self.assertIsInstance(self.metric.options, OrderedDict)
        self.assertGreater(len(self.metric.options), 0)

    def test_connection(self):
        self.assertIsInstance(self.metric.connection, pymongo.MongoClient)
        self.assertIsInstance(self.metric.db, pymongo.database.Database)
        self.assertIsInstance(self.metric.trainings,
                              pymongo.collection.Collection)
        self.assertIsInstance(self.metric.metrics,
                              pymongo.collection.Collection)
        self.assertIsInstance(self.metric.meta,
                              pymongo.collection.Collection)
        info = self.metric.connection.server_info()
        self.assertEqual(int(info['ok']), 1)

    def test_record(self):
        doc = {u'cost' : 0.22,
               u'validation_accuracy' : 0.9,
               u'epoch' : 1, u'iteration' : 4000, u'second' : 209 }
        self.metric.record(**doc) # fancy unpacking trick
        record = self.metric.metrics.find_one({'job_id' : 1})
        doc2 = record
        doc2.pop('_id', None)
        doc[u'job_id'] = 1
        doc[u'type'] = 'train'
        self.assertDictEqual(doc2, doc)

    def test_owntype_record(self):
        doc = {u'cost' : 0.22,
               u'validation_accuracy' : 0.9,
               u'epoch' : 1, u'iteration' : 4000, u'second' : 209,
               'type' : 'pretrain_1'}
        self.metric.record(**doc) # fancy unpacking trick
        record = self.metric.metrics.find_one({'job_id' : 1})
        doc2 = record
        doc2.pop('_id', None)
        doc[u'job_id'] = 1
        self.assertDictEqual(doc2, doc)

    def test_clean_job(self):
        doc1 = {u'cost' : 0.22,
               u'validation_accuracy' : 0.9,
               u'epoch' : 1, u'iteration' : 4000, u'second' : 209 }
        doc2 = {u'cost' : 0.26,
               u'validation_accuracy' : 0.89,
               u'epoch' : 2, u'iteration' : 8000, u'second' : 418 }
        self.metric.record(**doc1) # fancy unpacking trick
        self.metric.record(**doc2) # fancy unpacking trick

        doc1 = {u'job_id' : 2, u'cost' : 0.22,
               u'validation_accuracy' : 0.9,
               u'epoch' : 1, u'iteration' : 4000, u'second' : 209 }
        doc2 = {u'job_id' : 2, u'cost' : 0.26,
               u'validation_accuracy' : 0.89,
               u'epoch' : 2, u'iteration' : 8000, u'second' : 418 }
        self.metric.record(**doc1) # fancy unpacking trick
        self.metric.record(**doc2) # fancy unpacking trick

        ln = len([x for x in self.metric.metrics.find()])
        self.assertEqual(ln, 4)
        self.metric.clean_job(job_id=1)
        ln = len([x for x in self.metric.metrics.find()])
        self.assertEqual(ln, 2)
        ln = len([x for x in self.metric.metrics.find({'job_id' : 1})])
        self.assertEqual(ln, 0)

    def test_experiment_metainfo(self):
        constants = {
          u'mini_batch_size' : 500,
          u'batchsize' : 5000000,
          u'limit' : 20,
          u'epoch' : 100,
          u'patience' : 20000,
          u'patience_increase' : 2,
          u'improvement_threshold' : 0.995,
          u'validation_frequency' : 5000
        }
        self.metric.add_experiment_metainfo(constants=constants)
        experiment_name = self.metric.options['experiment-name']
        exp = self.metric.meta.find_one({'experiment_name' : experiment_name})
        self.assertDictEqual(exp['constants'], constants)

