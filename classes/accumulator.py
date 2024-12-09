import pandas as pd
import os

'''
    Implement a tool that stores temporary query-item user evaluations and write final evaluations on file
'''
class Accumulator:

    def __init__(self, db_retrieved, user_preferences, bootstrap_samples_evaluations):
        self.count = {}
        self.accumulated = {}
        self.list_values = {}
        self.user_preferences = user_preferences
        self.bootstrap_samples_evaluations = bootstrap_samples_evaluations
        self.initialize(db_retrieved)

    # Initialization with "empty" dictionaries
    def initialize(self, db_retrieved):
        for row in db_retrieved.itertuples():
            keys = row.items
            values = [0.0] * len(row.items)
            values_bootstrap = [[0.0] * self.bootstrap_samples_evaluations for _ in range(len(row.items))]
            self.count[row.query_id] = dict(zip(keys, values))
            self.accumulated[row.query_id] = dict(zip(keys, values))
            self.list_values[row.query_id] = dict(zip(keys, values_bootstrap))

    # Update the state of the dictionaries
    def update(self, queries_ids, items_ids, evaluations, b):
        for (query_id, item_id, evaluation) in zip(queries_ids, items_ids, evaluations):
            if evaluation > 0:
                self.count[query_id][item_id] += 1
                self.accumulated[query_id][item_id] += evaluation
                self.list_values[query_id][item_id][b] = evaluation

    # Save on file the evaluations for one user preference over all queries and items pairs
    def save_one(self, queries_ids, items_ids, path = './data_imdb/db_evaluations.csv'):
        avg_evaluations = []
        sum_evaluations = []
        count_evaluations = []
        list_values = []
        for (query_id, item_id) in zip(queries_ids, items_ids):
            try:
                avg_evaluations.append(self.accumulated[query_id][item_id] / self.count[query_id][item_id])
            except ZeroDivisionError:
                avg_evaluations.append(0.0)
            sum_evaluations.append(self.accumulated[query_id][item_id])
            count_evaluations.append(self.count[query_id][item_id])
            list_values.append(self.list_values[query_id][item_id])
        df = pd.DataFrame({'query_id' : queries_ids, 'item' : items_ids, 'user_preferences' : self.user_preferences, 'avg' : avg_evaluations, 'sum' : sum_evaluations, 'count' : count_evaluations, 'list_values' : list_values})
        df.to_csv(path, header = not os.path.exists(path), index = False, mode = 'a+')

    # Save on file all the evaluations for all user preferences
    def save(self, path = './data_imdb/db_evaluations.csv'):
        queries_ids = []
        items_ids = []
        avg_evaluations = []
        sum_evaluations = []
        count_evaluations = []
        list_values = []
        for query_id, inner_dict in self.accumulated.items():
            for item_id in inner_dict.keys():
                queries_ids.append(query_id)
                items_ids.append(item_id)
                try:
                    avg_evaluations.append(self.accumulated[query_id][item_id] / self.count[query_id][item_id])
                except ZeroDivisionError:
                    avg_evaluations.append(0.0)
                sum_evaluations.append(self.accumulated[query_id][item_id])
                count_evaluations.append(self.count[query_id][item_id])
                list_values.append(self.list_values[query_id][item_id])
        df = pd.DataFrame({'query_id' : queries_ids, 'item' : items_ids, 'user_preferences' : self.user_preferences, 'avg' : avg_evaluations, 'sum' : sum_evaluations, 'count' : count_evaluations, 'list_values' : list_values})
        df.to_csv(path, header = not os.path.exists(path), index = False, mode = 'a+')

    # Read the saved evaluations
    def read_evaluations(self, path = './data_imdb/db_evaluations.csv'):
        return pd.read_csv(path, index_col = False)