from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from itertools import chain
import random
import os
import numpy as np
import pandas as pd
from ast import literal_eval

'''
    Implement a retriever for retrieving items based on the similarity of their embeddings with the user query
'''
class Retriever:
    
    def __init__(self, dataset, api_key, index_name, dimension, metric):
        # Embedding model for the vector database
        self.embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.dimension = dimension
        # Item dataset
        self.dataset = dataset
        # Number of items
        self.dataset_length = len(dataset)
        # Pinecone client
        self.pc = Pinecone(api_key = api_key)
        # Create serverless index if not yet created
        if not self.pc.has_index(index_name):
            self.pc.create_index(
                name = index_name,
                dimension = dimension,
                metric = metric,
                spec = ServerlessSpec(cloud='aws', region='us-east-1') 
            )
        # Select the index
        self.index = self.pc.Index(index_name)

    # Insert embedded vectors into the index under the specified namespace
    def initialize_index(self, namespace):
        vectors = self.get_embedded_vectors()
        self.index.upsert(vectors = vectors, namespace = namespace)
    
    # Create and return the embeddings of each observation in the original dataset
    def get_embedded_vectors(self):
        vectors = []
        for id, row in self.dataset.iterrows():
            item = "{}: {} Directed by {}. Starring: {}, {}, {}, {}. Genre: {}".format(row['Series_Title'], row['Overview'], row['Director'], row['Star1'], row['Star2'], row['Star3'], row['Star4'], row['Genre'])
            item_embedding = self.embedding_model.get_text_embedding(item)
            vectors.append({"id" : str(id), "values" : item_embedding})
        return vectors
    
    # Return ids of k1 relevant items, k2 random items, and k3 random non-relevant items
    def retrieve_ids(self, query_id, query, namespace, k1, k2 = 0, k3 = 0, path = "./data_imdb/db_retrieved.csv"):
        retrieved_ids_1 = self.retrieve_top_k(query, k1, namespace)
        retrieved_ids_2 = self.retrieve_random_k(k2)
        retrieved_ids_3 = self.retrieve_rnr_k(query, k3, namespace)
        retrieved_ids = list(chain(retrieved_ids_1, retrieved_ids_2, retrieved_ids_3))
        true_lables = list(chain(["REL"]* k1, ["RAN"]* k2, ["RNR"]* k3))
        self.save_results(path, query_id, query, retrieved_ids, true_lables)
        return retrieved_ids, true_lables
    
    # Return json string of k1 relevant items, k2 random items, and k3 random non-relevant items
    def retrieve(self, retrieved_ids, shuffle = False):
        if shuffle:
            random.shuffle(retrieved_ids)
        retrieved_obs = self.get_retrieved_obs(retrieved_ids)
        return retrieved_obs.drop_duplicates().to_json(orient="index")
        
    # Return ids of relevant retrieved objects based on the query
    def retrieve_top_k(self, query, k, namespace):
        if k <= 0:
            return []
        embedded_query = self.embedding_model.get_query_embedding(query)
        results = self.index.query(namespace = namespace, vector = embedded_query, top_k = k, include_values = False, include_metadata = False)
        retrieved_ids = self.get_retrieved_ids(results)
        return retrieved_ids
    
    # Return ids of random non-relevant (rnr) retrieved objects based on the query
    def retrieve_rnr_k(self, query, k, namespace):
        if k <= 0:
            return []
        embedded_query = self.embedding_model.get_query_embedding(query)
        results = self.index.query(namespace = namespace, vector = embedded_query, top_k = 10*k, include_values = False, include_metadata = False)
        retrieved_ids = self.get_retrieved_ids(results)
        rnr_ids = random.choices(list(set(range(self.dataset_length)) - set(retrieved_ids)), k = k)
        return rnr_ids
    
    # Return ids of randomly retrieved objects
    def retrieve_random_k(self, k):
        if k <= 0:
            return []
        retrieved_ids = [random.randint(0, self.dataset_length-1) for _ in range(k)]
        return retrieved_ids

    # Return the ids of the retrieved results
    def get_retrieved_ids(self, results):
        retrieved_ids = []
        for result in results['matches']:
            retrieved_ids.append(int(result['id']))
        return retrieved_ids

    # Return the observations from the dataset corresponding to the retrieved ids
    def get_retrieved_obs(self, top_k_ids):
        return(self.dataset.iloc[top_k_ids])
    
    # Return the sequence of query-item pairs, queries' ids and items' ids
    def get_query_item_sequence(self, batch):
        queries = []
        queries_ids = []
        items = []
        items_ids = []
        count = 0
        for row in batch.itertuples():
            queries.append([row.query] * len(row.items))
            queries_ids.append([row.query_id] * len(row.items))
            count += len(row.items)
            items.append([self.retrieve(id, shuffle = False) for id in row.items])
            items_ids.append([id for id in row.items])
        random_permutation = np.random.default_rng().permutation(count)
        queries_permuted = np.array(list(chain.from_iterable(queries)))[random_permutation]
        queries_ids_permuted = np.array(list(chain.from_iterable(queries_ids)))[random_permutation]
        items_permuted = np.array(list(chain.from_iterable(items)))[random_permutation]
        items_ids_permuted = np.array(list(chain.from_iterable(items_ids)))[random_permutation]
        return([{"query" : query, "item" : item} for (query,item) in zip(queries_permuted,items_permuted)], queries_ids_permuted, items_ids_permuted)
    
    # Log the result in a file
    def save_results(self, path, query_id, query, retrieved_ids, true_labels):
        df = pd.DataFrame({'query_id': query_id,
                           'query' : query, 
                           'items' : [retrieved_ids], 
                           'labels' : [true_labels], 
                        })
        df.to_csv(path, header = not os.path.exists(path), index = False, mode = 'a')
    
    # Read the saved retreived items
    def read_saved_results(self, path):
        return pd.read_csv(path, index_col = False, converters={"items": literal_eval, "labels": literal_eval})