import numpy as np
import hashlib
import os
import pandas as pd
from openai import OpenAI

'''
    Implement an LLM user that creates queries and evaluates recommendations
'''
class User:
    
    def __init__(self, api_key, model_name):
        self.client = OpenAI(api_key = api_key)
        self.system_prompt = "You are a person who wants to watch a movie on an online streaming service."
        self.model_name = model_name

    # Return the query generated by the LLM
    def generate_queries(self, path = './data_imdb/db_queries.csv', temperature = 1.0):
        prompt = self.get_query_prompt()
        messages = [{"role": "system", "content": self.system_prompt}] + [{"role": "user", "content": prompt}]
        response = self.llm_request(messages, temperature)
        queries = response.choices[0].message.content.split('|')
        queries_ids = self.get_queries_ids(queries)
        self.save_queries(queries_ids, queries, path)

    # Return the evaluation of the recommended item on a 5-point Likert scale 
    def generate_evaluations(self, sequence, user_preferences = "no preferences", temperature = 1.0):
        prompt = self.get_evaluation_prompt(sequence, user_preferences)
        messages = [{"role": "system", "content": self.system_prompt}] + [{"role": "user", "content": prompt}]
        response = self.llm_request(messages, temperature)
        evaluations_str = response.choices[0].message.content.split()
        evaluations_int = []
        for eval in evaluations_str:
            try:
                int_eval = int(eval)
            except:
                int_eval = 0
            if int_eval < 1 or int_eval > 5:
                int_eval = 0
            evaluations_int.append(int_eval)
        return(evaluations_int)

    # Return the prompt used to ask the LLM to generate a query
    def get_query_prompt(self):
        prompt = '''
            Generate a few dozens queries in natural language to express what you are looking for on the platform.
            Do not ask for documentaries. Do not include temporal references (e.g. recent movies, movies from a particular year, movies from a particular period, etc.). Do not ask for popular or trending movies. Keep the queries relatively short.
            An example of generated queries is: I feel like watching an horror|Are there movies with Di Caprio?|Show me one by Tarantino|Films in New York|World War I|Something funny
            Please use the same format as the example. Do not include anything else in the output.
        '''
        return prompt
    
    def get_evaluation_prompt(self, sequence, user_preferences = "no preferences"):
        if user_preferences == "no preferences":
            prompt = '''
                You will be given a sequence of query-movie pairs in json format. Each movie is in json format. Given the query, assess the quality of the recommended movie using the following Likert scale:
                1: very poor quality; 2: poor quality; 3: acceptable quality; 4: good quality; 5: very good quality.

                The output should have the form: quality_1 quality_2 .. quality_K where quality_1 is the quality of the first query-movie pair, quality_1 is the quality of the second query-movie pair, and so on.
                The sequence does not have a particular order, so please evaluate each query-movie pair independently. Do not include anything else in the output.

                The sequence of query-movie pairs is: {sequence}
            '''
            return(prompt.format(sequence = sequence))
        else:
            prompt = '''
                You will be given a sequence of query-movie pairs in json. Each movie is in json format. 
                
                Your personal preferences are: {user_preferences}

                Given the query and your personal preferences, assess the quality of the recommended movie using the following Likert scale:
                1: very poor quality; 2: poor quality; 3: acceptable quality; 4: good quality; 5: very good quality.

                The output should have the form: quality_1 quality_2 .. quality_K where quality_1 is the quality of the first query-movie pair, quality_2 is the quality of the second query-movie pair, and so on.
                The sequence does not have a particular order, so please evaluate each query-movie pair independently. Do not include anything else in the output.

                The sequence of query-movie pairs is: {sequence}
            '''
            return(prompt.format(sequence = sequence, user_preferences = user_preferences))
    
    def get_evaluation_prompt_v1(self, sequence, user_preferences = "no preferences"):
        if user_preferences == "no preferences":
            prompt = '''
                You will be given a sequence of query-movie pairs in json format. Each movie is in json format. Express how likely you are to watch the movie using the following scale:
                1: very unlikely; 2: unlikely; 3: unsure; 4: likely; 5: very likely.

                The output should have the form: likelihood_1 likelihood_2 .. likelihood_K
                The sequence does not have a particular order, so please evaluate each query-movie pair independently. Do not include anything else in the output, only how likely you are to watch each movie using the 1-5 scale.

                The sequence of query-item pairs is: {sequence}
            '''
            return(prompt.format(sequence = sequence))
        else:
            prompt = '''
                You will be given a sequence of query-movie pairs in json format. Each movie is in json format. Express how likely you are to watch the movie using the following scale:
                1: very unlikely; 2: unlikely; 3: unsure; 4: likely; 5: very likely.

                The output should have the form: likelihood_1 likelihood_2 .. likelihood_K
                The sequence does not have a particular order, so please evaluate each query-movie pair independently. Do not include anything else in the output, only how likely you are to watch each movie using the 1-5 scale.
                
                When expressing how likely you are to watch the movie, conisder also your personal preferences: {user_preferences}
                The sequence of query-item pairs is: {sequence}
            '''
            return(prompt.format(sequence = sequence, user_preferences = user_preferences))

    # Prompt the LLM to generate the query
    def llm_request(self, messages, temperature):
        try:
            return self.client.chat.completions.create(model = self.model_name, messages = messages, temperature = temperature)
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e
        
    # Create a unique ID for each query
    def get_queries_ids(self, queries):
        queries_ids = []
        for query in queries:
            queries_ids.append(self.get_id(query))
        return queries_ids

    # Create an unique ID for a query (same queries will have the same ID)
    def get_id(self, query):
        h = hashlib.sha256()
        h.update(query.encode('utf-8'))
        return h.hexdigest()
        
    # Log the queries in a file
    def save_queries(self, queries_ids, queries, path = './data_imdb/db_queries.csv'):
        df = pd.DataFrame({'query_id' : queries_ids, 'query' : queries})
        df.to_csv(path, header = not os.path.exists(path), index = False, mode = 'a')

    # Read the saved queries
    def read_queries(self, path):
        return pd.read_csv(path, index_col = False)