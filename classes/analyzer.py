import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from matplotlib.lines import Line2D

'''
    Implement a class to analyze the results of the experiment
'''

class Analyzer:

    def __init__(self, dataset, db_rank, db_eval, all_user_preferences):
        self.dataset = dataset
        self.db_rank = db_rank
        self.db_eval = db_eval
        self.all_user_preferences = all_user_preferences
    
    # Return the observations from the dataset corresponding to a given entry, as a Pandas dataframe
    def get_observations(self, entry):
        ids = self.db_rank.iloc[entry]['ranking']
        return(self.dataset.iloc[ids])

    # Return the observations from the dataset corresponding to a given entry, in json format
    def get_observations_json(self, entry):
        ids = self.db_rank.iloc[entry]['ranking']
        return(self.dataset.iloc[ids].to_json(orient="index"))
    
    # Compute the mean MRR, DCG, and nDCG
    def compute_mean_ranking_metrics(self, user_preferences, K = None, threshold = 4.0):
        temp_db_rank = self.db_rank.loc[self.db_rank['user_preferences'] == user_preferences]
        mrr = np.mean(self.compute_mrr(user_preferences, temp_db_rank, threshold))
        dcg = np.mean(self.compute_dcg_K(user_preferences, temp_db_rank, K = K))
        ndcg = np.mean(self.compute_ndcg_K(user_preferences, temp_db_rank, K = K))
        b_ndcg = np.mean(self.compute_binary_ndcg_K(user_preferences, temp_db_rank, K = K, threshold = threshold))
        return({'MRR': mrr, 'DCG' : dcg, 'nDCG': ndcg, 'binary nDCG' : b_ndcg})
    
    # Plot the density estimation of each ranking metric, per user preference
    def plot_ranking_metrics(self, K = None, threshold = 4.0):
        data = self.get_plot_ranking_metrics_data(K, threshold)
        for metric_name in ['mrr', 'dcg', 'ndcg', 'binary_ndcg']:
            sns.displot(data = data, x = metric_name, hue = "user_preference", kind = "kde", rug = False)

    # Plot the density estimation of x@K, per user preference
    def plot_x_K(self, K = 0, save = False):
        data = self.get_plot_x_K_data(K)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # Shared y-axis for consistency
        colors = sns.color_palette("Pastel1", as_cmap=False)[:len(self.all_user_preferences )]          # Define a consistent color palette
        for ax, x, x_label in zip(axes, ['Runtime', 'Released_Year'], ['Runtime', 'Year of Release']):
            kde = sns.kdeplot(
                data=data, 
                x=x, 
                hue="user_preference", 
                ax=ax, 
                fill=False,       # Use filled curves
                palette=colors,  # Apply the defined color palette
                common_norm=False,  # Avoid normalization across hues
                legend=False
            )
            ax.set_ylabel(None) 
            ax.set_yticks([])  
            ax.set_xlabel(x_label, fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
        labels = ["No Preference", "Vintage Movies", "Short Movies", "Short and Vintage Movies"]
        fig.legend(lines, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        if save:
            folder = './plots_imdb/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig('{}/densities_x.png'.format(folder), dpi = 200, bbox_inches='tight')
        plt.show()

    # Plot the estimated regression coefficients per ranking position
    def plot_regression_coef(self, y_name, title = "", alpha = 0.05, K = 7, delta = 0.1, save = False, threshold = 4.0):
        reduced_user_preferences = self.all_user_preferences.copy()
        reduced_user_preferences.pop(0)
        x = [[] for _ in reduced_user_preferences]
        jitters = self.get_jitters(len(reduced_user_preferences), delta)
        estimated_coefficients = [[] for _ in reduced_user_preferences]
        confidence_intervals = [[[],[]] for _ in reduced_user_preferences]
        colors = sns.color_palette("Pastel1", as_cmap=False)[1:len(reduced_user_preferences)+1] 
        labels = ["Vintage Movies", "Short Movies", "Short and Vintage Movies"]
        fig = plt.figure()
        for k in range(K):
            results = self.regression(y_name, k, threshold)
            for i, user_preferences in enumerate(reduced_user_preferences):
                x[i].append(k+jitters[i])
                estimated_coefficients[i].append(results.params[i+1])
                for j in range(2):
                    confidence_intervals[i][j].append(results.conf_int(alpha = alpha, cols = None)[i+1][j])
        for i, _ in enumerate(reduced_user_preferences):
            plt.scatter(x[i], estimated_coefficients[i], marker = 'o', color = colors[i])
            plt.vlines(x[i], confidence_intervals[i][0], confidence_intervals[i][1], alpha = 0.5,  color = colors[i])
        plt.hlines(0, -1, K, colors = 'gainsboro', linestyles = '-', alpha = 0.3)
        plt.xlabel('Rank')
        plt.xticks(np.arange(K), np.arange(1,K+1))
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
        plt.legend(lines, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.25))
        plt.tight_layout()
        if save:
            folder = './plots_imdb/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig('{}/regression_coefficients_{}.png'.format(folder, y_name), dpi = 200, bbox_inches='tight')
        plt.show()
            
    # Return dataset to plot the ranking metrics
    def get_plot_ranking_metrics_data(self, K = None, threshold = 3.0):
        data = pd.DataFrame({'user_preference' : [], 'mrr' : [], 'dcg' : [], 'ndcg' : [], 'binary_ndcg' : []})
        for user_preferences in self.all_user_preferences:
            temp_db_rank = self.db_rank.loc[self.db_rank['user_preferences'] == user_preferences]
            mrr = self.compute_mrr(user_preferences, temp_db_rank, threshold)
            dcg = self.compute_dcg_K(user_preferences, temp_db_rank, K = K)
            ndcg = self.compute_ndcg_K(user_preferences, temp_db_rank, K = K)
            b_ndcg = self.compute_binary_ndcg_K(user_preferences, temp_db_rank, K, threshold)
            temp_data = pd.DataFrame({'user_preference' : [user_preferences] * temp_db_rank.shape[0], 'mrr' : mrr, 'dcg' : dcg, 'ndcg' : ndcg, 'binary_ndcg' : b_ndcg})
            data = pd.concat([data, temp_data])
        return(data)
    
    # Return dataset to plot the x@K
    def get_plot_x_K_data(self, K = 0):
        data = pd.DataFrame({'user_preference' : [], 'Runtime' : [], 'Released_Year' : []})
        for user_preferences in self.all_user_preferences:
            temp_db_rank = self.db_rank.loc[self.db_rank['user_preferences'] == user_preferences]
            runtime = np.zeros(temp_db_rank.shape[0])
            released_year = np.zeros(temp_db_rank.shape[0])
            for j, row in enumerate(temp_db_rank.itertuples()):
                runtime[j] = self.get_value_K('Runtime', row.ranking, K)
                released_year[j] = self.get_value_K('Released_Year', row.ranking, K)
            temp_data = pd.DataFrame({'user_preference' : [user_preferences] * temp_db_rank.shape[0], 'Runtime' : runtime, 'Released_Year' : released_year})
            data = pd.concat([data, temp_data])
        return(data)
        
    # Return the mean reciprocal rank for a given user preference
    def compute_mrr(self, user_preferences, temp_db_rank, threshold = 4.0, K = None):
        mrr = np.zeros(temp_db_rank.shape[0])
        for j, row in enumerate(temp_db_rank.itertuples()):
            ranked_evaluations = self.get_ranked_evaluations(user_preferences, row.query_id, row.ranking)
            for rank, eval in enumerate(ranked_evaluations):
                if eval >= threshold:
                    mrr[j] += 1/(rank+1)
        return mrr

    # Return the discounted cumulative gain at K for a given user preference
    def compute_dcg_K(self, user_preferences, temp_db_rank, K = None, threshold = None):
        dcg = np.zeros(temp_db_rank.shape[0])
        for j, row in enumerate(temp_db_rank.itertuples()):
            ranked_evaluations = self.get_ranked_evaluations(user_preferences, row.query_id, row.ranking)
            for rank, eval in enumerate(ranked_evaluations):
                dcg[j] += eval / np.log2(rank+2)
                if (K is not None) and (rank == K-1):
                    break
        return dcg
    
    def compute_binary_dcg_K(self, user_preferences, temp_db_rank, K = None, threshold = 4.0):
        dcg = np.zeros(temp_db_rank.shape[0])
        for j, row in enumerate(temp_db_rank.itertuples()):
            ranked_evaluations = self.get_ranked_evaluations(user_preferences, row.query_id, row.ranking)
            for rank, eval in enumerate(ranked_evaluations):
                rel = 1 if eval > threshold else 0
                dcg[j] += 2**rel / np.log2(rank+2)
                if (K is not None) and (rank == K-1):
                    break
        return dcg
    
    # Return the normalized discounted cumulative gain at K for a given user preference
    def compute_ndcg_K(self, user_preferences, temp_db_rank, K = None, threshold = None):
        ndcg = np.zeros(temp_db_rank.shape[0])
        dcg = self.compute_dcg_K(user_preferences, temp_db_rank, K)
        for j, row in enumerate(temp_db_rank.itertuples()):
            ranked_evaluations = self.get_ranked_evaluations(user_preferences, row.query_id, row.ranking)
            idcg = 0.0
            ranked_evaluations.sort(reverse = True)
            for rank, eval in enumerate(ranked_evaluations):
                idcg += eval / np.log2(rank+2)
                if (K is not None) and (rank == K-1):
                    break
            ndcg[j] += dcg[j]/idcg
        return ndcg
    
    # Return the normalized discounted cumulative gain at K for a given user preference
    def compute_binary_ndcg_K(self, user_preferences, temp_db_rank, K = None, threshold = 4.0):
        ndcg = np.zeros(temp_db_rank.shape[0])
        dcg = self.compute_binary_dcg_K(user_preferences, temp_db_rank, K, threshold)
        for j, row in enumerate(temp_db_rank.itertuples()):
            ranked_evaluations = self.get_ranked_evaluations(user_preferences, row.query_id, row.ranking)
            idcg = 0.0
            ranked_evaluations.sort(reverse = True)
            for rank, eval in enumerate(ranked_evaluations):
                rel = 1 if eval > threshold else 0
                idcg += 2**rel / np.log2(rank+2)
                if (K is not None) and (rank == K-1):
                    break
            ndcg[j] += dcg[j]/idcg
        return ndcg
    
    # Return the discounted cumulative gain at K w.r.t. a custom metric for a given user preference
    def compute_custom_dcg_K(self, user_preferences, metric_name, K = None):
        c1 = self.dataset[metric_name].max()
        c2 = c1 - self.dataset[metric_name].min()
        temp_db_rank = self.db_rank.loc[self.db_rank['user_preferences'] == user_preferences]
        custom_dcg = np.zeros(temp_db_rank.shape[0])
        for j, row in enumerate(temp_db_rank.itertuples()):
            ranked_metrics = self.get_ranked_metric(metric_name, row.ranking, c1, c2)
            for rank, metric in enumerate(ranked_metrics):
                custom_dcg[j] += metric / np.log2(rank+2)
                if (K is not None) and (rank == K-1):
                    break
        return custom_dcg
    
    # Run the regression
    def regression(self, y_name, K = 0, threshold = 3.0):
        y, X = self.get_regression_data(y_name, K, threshold)
        results = sm.OLS(y, X).fit()
        return results
    
    # Return the values to jitter the plot of the regression coefficients
    def get_jitters(self, N, delta):
        if N%2 == 0:
            upper = [(n+1)*delta for n in range(N//2)]
            lower = [(n-1)*(-delta) for n in range(1+N//2,1,-1)]
            result = lower + upper
        else:
            upper = [(n+1)*delta for n in range((N-1)//2)]
            lower = [(n-1)*(-delta) for n in range(1+(N-1)//2,1,-1)]
            result = lower + [0.0] + upper
        return (result)


    # Return the dataset for the regression
    def get_regression_data(self, y_name, K = 0, threshold = 3.0):
        y = []
        X = []
        if y_name == 'Runtime' or y_name == 'Released_Year':
            for user_preferences in self.all_user_preferences:
                temp_db_rank = self.db_rank.loc[self.db_rank['user_preferences'] == user_preferences]
                x = [1, user_preferences == 'vintage movies', user_preferences == 'short movies', user_preferences == 'short and vintage movies']
                for row in temp_db_rank.itertuples():
                    y.append(self.get_value_K(y_name, row.ranking, K))
                    X.append(x)
        elif y_name == 'mrr' or y_name == 'dcg' or y_name == 'ndcg' or y_name == 'binary_ndcg':
            if y_name == 'mrr':
                f = getattr(Analyzer, 'compute_mrr')
            else:
                f = getattr(Analyzer, 'compute_{}_K'.format(y_name))
            for user_preferences in self.all_user_preferences:
                temp_db_rank = self.db_rank.loc[self.db_rank['user_preferences'] == user_preferences]
                x = [1, user_preferences == 'vintage movies', user_preferences == 'short movies', user_preferences == 'short and vintage movies']
                y_temp = f(self, user_preferences, temp_db_rank, K = K, threshold = threshold)
                for el in y_temp:
                    y.append(el)
                    X.append(x)
        elif y_name == 'rnr_ratio':
            for user_preferences in self.all_user_preferences:
                temp_db_rank = self.db_rank.loc[self.db_rank['user_preferences'] == user_preferences]
                x = [1, user_preferences == 'vintage movies', user_preferences == 'short movies', user_preferences == 'short and vintage movies']
                for row in temp_db_rank.itertuples():
                    y.append(self.get_rnr_K(row.labels, K))
                    X.append(x)
        return y, X
        
    # Return the permutation from retrieved ids to ranked ids
    def get_permutation_ids(self, retrieved_ids, ranked_ids):
        ids = []
        for item in ranked_ids:
            ids.append(retrieved_ids.index(item))
        return ids
    
    # Return the evaluations ordered according to the ranking
    def get_ranked_evaluations(self, user_preferences, query_id, ranked_items):
        temp_db_eval = self.db_eval.loc[self.db_eval['user_preferences'] == user_preferences]
        evaluations = []
        for item in ranked_items:
            condition = (temp_db_eval['query_id'] == query_id) & (temp_db_eval['item'] == item)
            evaluations.append(temp_db_eval[condition]['avg'].values[0])
        return(evaluations)
    
    # Return the metric ordered according to the ranking
    def get_ranked_metric(self, metric_name, ranked_items, c1, c2):
        metrics = []
        for item in ranked_items:
            metrics.append((c1-self.dataset.iloc[item][metric_name])/c2)
        return(metrics)
    
    # Return the value for the metric for the item in position K
    def get_value_K(self, metric_name, ranked_items, K = 0):
        for rank, item in enumerate(ranked_items):
            if rank == K:
                return(self.dataset.iloc[item][metric_name])
    
    # Return the proportion of random-non-relevant items ranked before position K
    def get_rnr_K(self, labels, K = 0):
        rnr = 0
        for rank, label in enumerate(labels):
            if label == 'RNR':
                rnr += 1
            if rank == K:
                break
        return(rnr/(K+1))
