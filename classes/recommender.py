import pandas as pd
import numpy as np
from ast import literal_eval
import os

'''
    Implement a retrieve-and-rank recommender system
'''
class Recommender:

    def __init__(self, namespace, retriever, ranker, bootstrap_samples = 1):
        self.namespace = namespace
        self.retriever = retriever
        self.ranker = ranker
        self.bootstrap_samples = bootstrap_samples
        
    # Return the ranked ids of the items given the query and retrieved items
    def rank_only(self, query_id, query, retrieved_ids, true_labels, k, user_preferences = "no preferences", path = "./data_imdb/db_ranking.csv", temperature = 1.0):
        copy_retrieved_ids = retrieved_ids.copy()
        dict_ids = dict(zip(copy_retrieved_ids, range(k)))
        rankings = np.empty((self.bootstrap_samples, k), dtype = np.uint64)
        valid_rankings = 0
        for b in range(self.bootstrap_samples):
            retrieved_items = self.retriever.retrieve(copy_retrieved_ids, True)
            ranking = self.ranker.rank(query, retrieved_items, user_preferences, temperature)
            r = ranking.copy()
            # If the ranked ids coincide with the retrieved ids, add them to the ranking
            if len(copy_retrieved_ids) == len(r):
                if set(copy_retrieved_ids) == set(r):
                    rankings[b] = ranking
                    valid_rankings += 1
        # If all the rankings are valid
        if valid_rankings == self.bootstrap_samples:
            final_ranking = self.get_interleaved_ranking(rankings, dict_ids)
            permutation = self.get_permutation_ids(retrieved_ids, final_ranking)
            self.save_ranking(path, query_id, query, final_ranking, list(np.array(true_labels)[permutation]), user_preferences)
            return final_ranking
        else:
            return None
    
    # Return a unified ranking list based on the bootstrapped rankings
    def get_interleaved_ranking(self, rankings, dict_ids):
        if self.bootstrap_samples == 1:
            return rankings[0].tolist()
        else:
            return self.borda_voting_rule(rankings, dict_ids)
        
    # Implement the Borda positional voting rule to interleave rankings
    def borda_voting_rule(self, rankings, dict_ids):
        k = rankings.shape[1]
        item_count = np.zeros(k)
        for b in range(self.bootstrap_samples):
            for pos, item in enumerate(rankings[b]):
                item_count[dict_ids[item]] += k-pos-1
        reverse_dict_ids = self.get_reverse_dict(dict_ids)
        ranked_keys = np.argsort(-item_count)
        ranked_items = []
        for key in ranked_keys:
            ranked_items.append(int(reverse_dict_ids[key]))
        return ranked_items
    
    # Return a dictionary with keys and values swapped
    def get_reverse_dict(self, dict_ids):
        reverse_dict = dict()
        for (key, val) in dict_ids.items():
            reverse_dict[val] = key
        return(reverse_dict)
    
    # Return the permutation from retrieved ids to ranked ids
    def get_permutation_ids(self, retrieved_ids, ranked_ids):
        ids = []
        for item in ranked_ids:
            ids.append(retrieved_ids.index(item))
        return ids

    # Log the result in a file
    def save_ranking(self, path, query_id, query, final_ranking, true_labels, user_preferences):
        df = pd.DataFrame({'query_id': query_id,
                           'query' : query, 
                           'ranking' : [final_ranking],
                           'labels' : [true_labels],
                           'user_preferences' : user_preferences,
                           'bootstrap_samples' : self.bootstrap_samples
                        })
        df.to_csv(path, header = not os.path.exists(path), index = False, mode = 'a+')

    # Read the saved retreived items
    def read_saved_results(self, path):
        return pd.read_csv(path, index_col = False, converters={"ranking": literal_eval, "labels": literal_eval})
