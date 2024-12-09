from openai import OpenAI

'''
    Implement an LLM listwise prompting ranker given the user query and the retrieved items
'''
class Ranker:
    
    def __init__(self, api_key, model_name):
        self.client = OpenAI(api_key = api_key)
        self.system_prompt = "You are an expert in ranking movies based on user queries about what they would like to watch."
        self.model_name = model_name

    # Return the ranked ids of the items based on the query
    def rank(self, query, retrieved_items, user_preferences = "no preferences", temperature = 1.0):
        prompt = self.get_rank_prompt(query, retrieved_items, user_preferences)
        messages = [{"role": "system", "content": self.system_prompt}] + [{"role": "user", "content": prompt}]
        response = self.llm_request(messages, temperature)
        try:
            ranked_ids = list(map(int, response.choices[0].message.content.split()))
        except:
            ranked_ids = []
        return ranked_ids
    
    # Return the prompt used to ask the LLM to rank the retrieved items
    def get_rank_prompt(self, query, retrieved_items, user_preferences):
        if user_preferences == "no preferences":
            prompt = '''
                You will be given a query entered by a user, and some retrieved movies to rank. Each movie is in json format. Please rank the movies based on their relevance to the user query.
            
                The output should have the form: ID_1 ID_2 .. ID_K where ID_1 is the ID of the most relevant movie, ID_2 is the ID of the second most relevant movie, and so on.
                Your output must include all the retrieved movies' IDs. Do not include anything else in the output.

                The user entered the query: {query}

                The retrieved movies are: {retrieved_items}
            '''
        else:
            prompt = '''
                You will be given a query entered by a user, the general user preferences for movies, and some retrieved movies to rank. Each movie is in json format.
                Please rank the movies based on their relevance to the user query while keeping into account the user preferences. Try your best to balance the different criteria.
            
                The output should have the form: ID_1 ID_2 .. ID_K where ID_1 is the ID of the best movie given the query and user preferences, ID_2 is the ID of the second best movie, and so on.
                Your output must include all the retrieved movies' IDs. Do not include anything else in the output.

                The user preferences are: {user_preferences}

                The user entered the query: {query}

                The retrieved movies are: {retrieved_items}
            '''
        return(prompt.format(query = query, retrieved_items = retrieved_items, user_preferences = user_preferences))

    # Return the prompt used to ask the LLM to rank the retrieved items
    def get_rank_prompt_v1(self, query, retrieved_items, user_preferences):
        if user_preferences == "no preferences":
            prompt = '''
                You will be given a query entered by a user, and some retrieved movies to rank. Each movie is in json format.
                Please rank the movies based on what the user may prefer to watch.
            
                The output should have the form: ID_1 ID_2 .. ID_K
                Your output must include all the retrieved movies' IDs. Do not include anything else in the output.

                The user entered the query: {query}

                The retrieved movies are: {retrieved_items}
            '''
        else:
            prompt = '''
                You will be given a query entered by a user, the general user preferences for movies, and some retrieved movies to rank. Each movie is in json format.
                Please rank the movies based on what the user may prefer to watch.
            
                The output should have the form: ID_1 ID_2 .. ID_K
                Your output must include all the retrieved movies' IDs. Do not include anything else in the output.

                The user entered the query: {query}

                The user preferences are: {user_preferences}

                The retrieved movies are: {retrieved_items}
            '''
        return(prompt.format(query = query, retrieved_items = retrieved_items, user_preferences = user_preferences))
    
    # Prompt the LLM to rank the retrieved items
    def llm_request(self, messages, temperature = 1.0):
        try:
            return self.client.chat.completions.create(model = self.model_name, messages = messages, temperature = temperature)
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e