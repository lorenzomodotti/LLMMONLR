from pandas import read_csv

'''
    Configuration of global parameters and variables
'''

# Set API keys
API_KEY_pinecone = ""
API_KEY_openai = ""

# Set OpenAI model to use as LLM
model_name = "gpt-4o-mini"

# Import and process IMDb dataset
dataset = read_csv("data_imdb/imdb_top_1000.csv", converters={"Released_Year": float}).drop(columns = ['Poster_Link', 'Certificate', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross'])
dataset['Runtime'] = list(map(lambda t: float(t.split()[0]), dataset['Runtime'].values))
print('> Movie dataset imported and processed')

# Specify user preferences to experiment with
all_user_preferences = ["no preferences", "vintage movies", "short movies", "short and vintage movies"]

# Path database queries
path_queries = "./data_imdb/db_queries.csv"
# Path database retrieved items
path_retrieved = "./data_imdb/db_retrieved.csv"
# Path database evaluations
path_evaluations = "./data_imdb/db_evaluations.csv"
# Path database rankings
path_ranking = "./data_imdb/db_ranking.csv"

# Name of Pinecone index
index_name = "project"
# Dimension of embedding
dimension = 384
# Metric to compute similarity
metric = "cosine"

# Number of relevant items to retrieve for each query
k1 = 5
# Number of random items to retrieve for each query
k2 = 0
# Number of non-relevant items to retrieve for each query
k3 = 3
# Total number of retrieved items
k = k1 + k2 + k3

# Namespace of the index
namespace = 'imdb'

# Number of bootstrap samples to compute evaluations
bootstrap_samples_evaluations = 3

# Number of bootstrap samples to compute rankings
bootstrap_samples_ranking = 3