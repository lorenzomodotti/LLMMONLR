import numpy as np
from tqdm import tqdm
from classes.user import User
from classes.retriever import Retriever
from classes.ranker import Ranker
from classes.recommender import Recommender
from classes.accumulator import Accumulator
from config import *

'''
    Run an experiment by: generating queries, retrieving items, evaluating query-item pairs, ranking retrieved items
'''

# Initialize user
user = User(API_KEY_openai, model_name)

# Generate queries by prompting the LLM
for _ in tqdm(range(10), desc = '> Prompting LLM to generate queries'):
    user.generate_queries(path = path_queries, temperature = 1.4)
print('> Queries created and saved on file')

# Read generated queries
db_queries = user.read_csv(path_queries)
db_queries.drop_duplicates(subset = ['query_id'], inplace = True)
print('> Queries dataset imported')

# Initialize retriever
retriever = Retriever(dataset, API_KEY_pinecone, index_name, dimension, metric)
print('> Retriever initialized')

# Initialize index
retriever.initialize_index(namespace)
print('> Index initialized')

# Retrieve items for each query and save the results
for row in tqdm(db_queries.itertuples(), desc = '> Querying Pinecone index to retrieve items'):
    retriever.retrieve_ids(row.query_id, row.query, namespace, k1, k2, k3, path_retrieved)
print('> Documents retrieved and saved on file')

# Read retrieved items
db_retrieved = retriever.read_saved_results(path_retrieved)
db_retrieved.drop_duplicates(subset = ['query_id'], inplace = True)
print('> Retrieved items dataset imported')

# Initialize accumulators
accumulators = {}
for user_preferences in all_user_preferences:
    accumulators[user_preferences] = Accumulator(db_retrieved, user_preferences, bootstrap_samples_evaluations)
print('> Accumulators initialized')

# Generate query-item evaluations by prompting the LLM
for batch in tqdm(np.array_split(db_retrieved, 100), desc = '> Prompting LLM to generate evaluations'):
    query_item_sequence, queries_ids, items_ids = retriever.get_query_item_sequence(batch)
    for user_preferences in all_user_preferences:
        for b in range(bootstrap_samples_evaluations):
            evaluations = user.generate_evaluations(query_item_sequence, user_preferences = user_preferences, temperature = 1.0)
            accumulators[user_preferences].update(queries_ids, items_ids, evaluations, b)
        accumulators[user_preferences].save_one(queries_ids, items_ids, path_evaluations)
print('> Evaluations saved on file')

# Read query-item evaluations
db_evaluations = accumulators[all_user_preferences[0]].read_evaluations(path_evaluations)
print('> Evaluations dataset imported')

# Initialize ranker
ranker = Ranker(API_KEY_openai, model_name)
print('> Ranker initialized')

# Initialize recommender
recommender = Recommender(namespace, retriever, ranker, bootstrap_samples_ranking)
print('> Recommender initialized')

# Generate rankings
for row in tqdm(db_retrieved.itertuples(), '> Prompting LLM to generate rankings'):
    query_id = row.query_id
    query = row.query
    retrieved_items = row.items
    true_labels = row.labels
    for user_preferences in all_user_preferences:
        recommender.rank_only(query_id, query, retrieved_items, true_labels, k, user_preferences = user_preferences, path = path_ranking)
print('> Rankings saved on file')