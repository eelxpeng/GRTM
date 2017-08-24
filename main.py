import pickle
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ptm import RelationalTopicModel
from ptm.utils import convert_cnt_to_list, get_top_words
import ptm.utility as utility

logger = logging.getLogger('RelationalTopicModel')
logger.propagate=False

userList, link_matrix = utility.init_data('./data/user.txt','./data/train.txt')
userList, test_link_matrix = utility.init_data('./data/user.txt','./data/test.txt')
doc_dir = './data/images/'
print len(userList)
print link_matrix.shape

n_doc = len(userList)
n_topic = 100
max_iter = 100

model = RelationalTopicModel(n_topic, n_doc, verbose=True)
model.fit(userList, doc_dir, link_matrix, test_link_matrix, max_iter=max_iter)