from classes.retriever import Retriever
from classes.ranker import Ranker
from classes.recommender import Recommender
from classes.accumulator import Accumulator
from classes.analyzer import Analyzer
from config import *

'''
    Analyze the results of an experiment by running the regression and displaying the plots
'''

# Initialize retriever
retriever = Retriever(dataset, API_KEY_pinecone, index_name, dimension, metric)
print('> Retriever initialized')

# Read retrieved items
db_retrieved = retriever.read_saved_results(path_retrieved)
db_retrieved.drop_duplicates(subset = ['query_id'], inplace = True)
print('> Retrieved items dataset imported')

# Initialize accumulators
accumulators = {}
for user_preferences in all_user_preferences:
    accumulators[user_preferences] = Accumulator(db_retrieved, user_preferences, bootstrap_samples_evaluations)
print('> Accumulators initialized')

# Read query-item evaluations
db_evaluations = accumulators[all_user_preferences[0]].read_evaluations(path_evaluations)
print('> Evaluations dataset imported')

# Initialize ranker
ranker = Ranker(API_KEY_openai, model_name)
print('> Ranker initialized')

# Initialize recommender
recommender = Recommender(namespace, retriever, ranker, bootstrap_samples_ranking)
print('> Recommender initialized')

# Read rankings
db_rank = recommender.read_saved_results(path_ranking)
print('> Ranked items dataset imported')

# Initialize analyzer
analyzer = Analyzer(dataset, db_rank, db_evaluations, all_user_preferences)

# Compute ranking metrics
for user_preferences in all_user_preferences:
    print(user_preferences)
    for key, value in analyzer.compute_mean_ranking_metrics(user_preferences, threshold = 3.0).items():
        print(' > ', key, ":", value)

# Plot distribution of ranking metrics at rank 0
analyzer.plot_ranking_metrics(K = 0)

# Plot distribution of movie attributes at rank 0
analyzer.plot_x_K(K = 0, save = False)

# Plot regression coefficient estimates with confidence intervals for each metric
for metric in ['ndcg', 'binary_ndcg', 'Runtime', 'Released_Year', 'rnr_ratio']:
    analyzer.plot_regression_coef(metric, K = 5, save = False)
