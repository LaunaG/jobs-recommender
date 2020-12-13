'''
cf_client.py

Methods for completing collaborative filtering and analysis of
the 2012 Kaggle Jobs Recommendation dataset.
'''
import glob
import gzip
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import random
import requests
import zipfile

from log import Log
from sklearn import metrics
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from IPython.core.display import display, HTML

class JobAppDatasets:

    ## DATASET LOADING ##
    def __init__(self):
        '''
        The constructor for the JobAppDatasets class. Loads the TSV data files
        into Pandas DataFrames from a local "data" folder.
        '''
        if not os.path.isdir("data"):
            self.load_data_from_storage()

        # Initialize logger
        self.log = Log()

        # Load users
        self.users = pd.read_csv("data/users.tsv", sep="\t")
        self.assign_user_groups()
        self.log.info("users.tsv", "h4")
        self.log.info("Holds all users and their metadata", "p")
        display(self.users.head(2))

        # Load apps
        self.apps = pd.read_csv("data/apps.tsv", sep="\t")
        self.log.info("apps.tsv", "h4")
        self.log.info("Holds the applications users submitted", "p")
        display(self.apps.head(2))

        # Load jobs assigned to first window
        self.jobs1 = pd.read_csv("data/jobs1.tsv", sep="\t")
        self.log.info("jobs1.tsv", "h4")
        self.log.info("Holds the jobs available on CareerBuilder.com during a 13-day window", "p")
        display(self.jobs1.head(2))

        # Load user history
        self.user_history = pd.read_csv("data/user_history.tsv", sep="\t")
        self.log.info("user_history.tsv", "h4")
        self.log.info("Holds users' past job title(s)", "p")
        display(self.user_history.head(2))

        # Load window dates
        self.window_dates = pd.read_csv("data/window_dates.tsv", sep="\t")
        self.log.info("window_dates.tsv", "h4")
        self.log.info("Holds the application window dates", "p")
        display(self.window_dates)


    def load_data_from_storage(self):
        '''
        Retrieves the data files from a remote storage endpoint and
        writes them to a local folder named "data".

        Parameters:
            None

        Returns:
            None
        '''
        # Retrieve zip file from Dropbox and write to base/default folder
        r = requests.get("https://www.dropbox.com/s/v2fdobitjrjieku/data.zip?dl=1")
        with open("data.zip", 'wb') as f:
            f.write(r.content)

        # Extract zip file contents to create local data folder with .tsv.gz files
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

        # For each unzipped file path
        for path in glob.glob("data/*.tsv.gz"):

            # Create destination file path
            dest_path = f'data/{os.path.basename(path)[:-3]}'

            # Open unzipped file for reading and destination file for writing
            with open(path, 'rb') as f:
                with open(dest_path, 'wb') as g:

                        # Decompress unzipped file data and write to destination
                        decompressed = gzip.decompress(f.read())
                        g.write(decompressed)

            # Delete original compressed file
            os.remove(path)

        # Delete zip file
        os.remove("data.zip")


    def assign_user_groups(self):
        '''
        Assigns each user to a user group.
        '''
        def assign_to_educ_group(degree_type):

            if degree_type in ["Associate's", "Bachelor's", "Vocational"]:
                return "College"

            elif degree_type in ["Master's", "PhD"]:
                return "Post-Graduate"

            else:
                return degree_type

        self.users["Group"] = (self
            .users["DegreeType"]
            .apply(lambda d: assign_to_educ_group(d)))


    ## DATA SUMMARY AND VISUALIZATION ##
    def preview_csr_matrix(self, csr):
        '''
        Displays a portion of the CSR matrix.

        Parameters:
            csr (scipy.sparse.csr_matrix): the matrix

        Returns:
            None
        '''
        fig = plt.figure(figsize=(50, 50))
        tick_range = np.arange(0, 80000, 100)
        plt.yticks(tick_range, list(tick_range))

        plt.xlabel("JobID")
        plt.ylabel("UserID")
        plt.spy(csr, markersize = 1, origin="lower")


    def summarize_csr_matrix(self, csr):
        '''
        Summarizes the CSR matrix.

        Parameters:
            csr (scipy.sparse.csr_matrix): the matrix

        Returns:
            None
        '''
        num_rows, num_cols = csr.shape
        num_cells = num_rows * num_cols
        num_entries = csr.nnz # here, the number of apps
        num_zeroes = num_cells - num_entries 
        sparsity = (num_zeroes / num_cells) * 100

        display(HTML(f"<h5>The CSR matrix has {num_rows:,} rows and " +
            f"{num_cols:,} columns with {num_cells:,} matrix cells total.</h4>"))
            
        display(HTML(f"<h5>{num_zeroes:,} of those elements are zeroes, so " + 
            f"the matrix is ~{sparsity} percent sparse.</h5>"))


    ## DATA RETRIEVAL ##
    def get_jobs_dataset(self, window_id):
        '''
        Retrieves the job dataset corresponding to the given window id.

        Parameters:
            window_id (int): the window id

        Returns:
            (pd.DataFrame): the jobs DataFrame
        '''
        if window_id == 1:
            return self.jobs1
        else:
            return None


    def get_window_date_range(self, window_id):
        '''
        Generates a DataFrame representing the date range of the application window.
        
        Parameters:
            window_id (int): the window id
            
        Returns:
            pd.DataFrame: The date range DataFrame. Contains one column, "Date",
                        whose entries are formatted as "yyyy-MM-dd," as well as
                        an index containing the same values.
        '''
        test_start = self.window_dates.query("Window == @window_id")["Train Start"][0]
        test_end = self.window_dates.query("Window == @window_id")["Test End"][0]
        return (pd
            .date_range(start=test_start, end=test_end, freq="D")
            .to_frame()
            .rename(columns={0: "Date"}))


    ## CREATION OF TRAIN AND TEST SETS ##
    def create_csr_matrix(
        self, 
        app_ids, 
        user_lookup, 
        job_lookup, 
        users_as_rows, 
        log):
        '''
        Creates a CSR ("Compressed Sparse Row") matrix.

        Parameters:
            app_ids (pd.DataFrame): the UserID-JobID pairs related to an app
            user_lookup (dict<int, int>): a lookup of user index by UserID
            job_lookup (dict<int, int>): a lookup of job index by JobID
            users_as_rows (bool): whether the matrix should have users as
                                  rows and jobs as columns

        Returns:
            (scipy.sparse.csr_matrix): the sparse matrix
        '''
        users = app_ids["UserID"].apply(lambda id: user_lookup[id]).tolist()
        jobs = app_ids["JobID"].apply(lambda id: job_lookup[id]).tolist()

        data = [1] * len(app_ids)
        data_rows = users if users_as_rows else jobs
        data_columns = jobs if users_as_rows else users
        num_rows = len(user_lookup) if users_as_rows else len(job_lookup)
        num_cols = len(job_lookup) if users_as_rows else len(user_lookup)

        log.info(f"Data Length: {len(data)}", "p")

        csr = csr_matrix(
            arg1=(data, (data_rows, data_columns)),
            shape=(num_rows, num_cols), 
            dtype=np.int8)

        return csr


    def create_train_and_test_sets(
        self,
        window_id, 
        users_as_rows=True,
        app_threshold=5,
        subset_pct=0.5,
        pct_test=0.2, 
        seed=1,
        random_mask=True,
        log_disabled=False):
        '''
        Creates train and test CSR matrices. The matrices are identical
        except that the train matrix has masked (i.e., 0) values for test user
        applications submitted during the testing portion of the
        application window.

        Parameters:
            window_id (int): the window id

        Returns:
            (scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, pd.DataFrame): 
                a tuple of three elements: (1) a CSR matrix representing the
                training set; (2) a CSR matrix reprsenting the testing set; and
                (3) a Pandas DataFrame holding user and job ids of the
                applications that were masked.
        '''
        log = Log(log_disabled)

        log.header(f"Preparing data for application Window {window_id} "
            "train and test sets...")

        # Restrict applications to users meeting app submission threshold
        log.header("Filtering data to only include most popular jobs and users (i.e., " 
            f"those sending or receiving {app_threshold} or more applications "
            "in the given window)")
        app_ids = self.filter_applications(window_id, app_threshold, log)
        log.detail(f"{len(app_ids)} apps after filtering")

        # Select a random subset of app ids for training
        log.header(f"Selecting random {subset_pct * 100} percent "
            "of these apps for testing...")
        random.seed(seed * 2)
        num_samples = int(np.ceil(subset_pct * len(app_ids)))
        population = app_ids.index.tolist()
        samples = random.sample(population, num_samples)
        app_ids = app_ids.loc[app_ids.index.isin(samples)]
        log.detail(f"{len(app_ids)} apps chosen by random selection")

        # Create id-index lookup for the qualified users
        user_ids = app_ids["UserID"].sort_values().unique().tolist()
        user_lookup = {val: idx for idx, val in enumerate(user_ids)}
        log.detail(f"Users successfully filtered. {len(user_ids)} users "
            f"submitted {app_threshold} or more apps in the window.")

        # Create id-index lookup for jobs in current window
        job_ids = app_ids["JobID"].sort_values().unique().tolist()
        job_lookup = {val: idx for idx, val in enumerate(job_ids)}
        log.detail("Jobs successfully filtered. The qualified users submitted "
            f"apps to {len(job_ids)} different qualified jobs in the window.")
        log.detail("Filtering complete")

        # Mask a percentage of applications for testing
        log.header("Masking applications to use as test cells...")
        masked_ids = self.mask_applications(app_ids, pct_test, seed, random_mask, log)
        log.detail(f"{len(masked_ids)} app ids successfully masked with zeroes.")

        # Finalize application ids used for training and test sets
        log.header("Finalizing application ids to use in train and test sets...")
        train_app_ids = (app_ids
            .merge(masked_ids, on=["UserID", "JobID"], how="outer", indicator=True)
            .query("_merge == 'left_only'")[["UserID", "JobID"]])
        test_app_ids = app_ids
        log.header("Data preparation complete")

        # Create training and testing matrices
        log.header("Creating train and test sets...")
        log.detail("Creating CSR train matrix...")
        train_matrix = self.create_csr_matrix(
            train_app_ids, 
            user_lookup, 
            job_lookup, 
            users_as_rows,
            log)

        log.detail("Creating CSR test matrix...")
        test_matrix = self.create_csr_matrix(
            test_app_ids,
            user_lookup,
            job_lookup,
            users_as_rows,
            log)

        log.detail("Training and testing sets successfully created")

        return (train_matrix, test_matrix, masked_ids, user_lookup, job_lookup)


    def filter_applications(self, window_id, app_threshold, log):
        '''
        Retrieves all job applications submitted during a given window and then
        filters them to include only those from users who submitted a minimum 
        "threshold" number of apps (e.g., 2 or more, 5 or more, etc.).
        
        Parameters:
            window_id (int): the window id
            app_threshold (int): the number of apps a user should submit to
                                 be included in the train or test datasets
        
        Returns:
            (pd.DataFrame): the filtered job application DataFrame. Contains
                            only "UserID" and "JobID" columns.
        '''
        # Filter job application DataFrame to show app counts by user
        app_counts_by_user = (self.apps
            .query("WindowID == @window_id")
            .groupby("UserID")
            .size()
            .to_frame()
            .rename(columns={0:"count"})
            .query("count >= @app_threshold"))

        # Display app counts for improved troubleshooting 
        app_counts_for_display = (app_counts_by_user["count"]
            .sort_values(ascending=False)
            .to_frame())

        log.dataframe(app_counts_for_display)

        # Filter job application DataFrame to show app counts by job
        app_counts_by_job = (self.apps
            .query("WindowID == @window_id")
            .groupby("JobID")
            .size()
            .to_frame()
            .rename(columns={0:"count"})
            .query("count >= @app_threshold"))

        # Display app counts for improved troubleshooting 
        app_counts_for_display = (app_counts_by_job["count"]
            .sort_values(ascending=False)
            .to_frame())

        log.dataframe(app_counts_for_display)

        # Return app ids belonging to qualified users meeting threshold
        qualified_users = app_counts_by_user.index.tolist()
        qualified_jobs = app_counts_by_job.index.tolist()
        query = "UserID in @qualified_users & JobID in @qualified_jobs"
        app_ids = self.apps.query(query)[["UserID", "JobID"]]

        return app_ids

    
    def mask_applications(self, app_ids, pct_test, seed, random_mask, log):
        '''
        Masks applications according to one of two strategies:
        (1) at random for a certain given percentage or (2) targeting only
        test users' applications in the test window.

        Parameters:
            app_ids (pd.DataFrame): the UserID and JobID columns comprising
                                    the application ids
            pct_test (float): the percentage of apps to assign to the test
                              group. Only relevant if "random_mask" is True.
            seed (int): the seed for the random generator. Only relevant if
                        "random_mask" is True.
            random_mask (bool): whether the random masking strategy
                                should be used.

        Returns:
            (pd.DataFrame): a subset of the app_ids DataFrame indicating
                            applications that should be masked
        '''
        if random_mask:

            log.detail("Randomly masking applications in dataset...")

            # Initialize random seed
            random.seed(seed)
            log.detail(f"Initializing random generator with seed {seed}...")

            # Initialize number of samples and population to draw from
            num_samples = int(np.ceil(pct_test * len(app_ids)))
            population = app_ids.index.tolist()
            log.detail(f"Reading configuration setting...{pct_test * 100} "
                "percent of apps should be used for testing out of the "
                f"{len(population)} total available...")

            # Sample from population of indices
            samples = random.sample(population, num_samples)
            log.detail(f"{len(samples)} app ids randomly chosen for masking...")

            # Mask apps corresponding to sampled/selected index values
            masked_ids = app_ids.loc[app_ids.index.isin(samples)]

        else:
            log.detail("Masking apps made by test users in test window...")

            # Determine date range for testing window
            window_start = (self.window_dates
                .query("Window == @window_id")["Test Start"][0])
            window_end = (self.window_dates
                .query("Window == @window_id")["Test End"][0])
            log.detail(f"The test window is {window_start} to {window_end}")

            # Filter apps to only include those in testing window
            masked_ids_query = (
                "WindowID == @window_id & " + 
                "Split == 'Test' & " + 
                "ApplicationDate >= @window_start & " + 
                "ApplicationDate <= @window_end")
            masked_ids = self.apps.query(masked_ids_query)[["UserID", "JobID"]]

        return masked_ids


    ## CROSS VALIDATION ##
    def perform_nmf(
        self, 
        csr, 
        k, 
        init="nndsvd", 
        max_iter=500, 
        random_state=0, 
        alpha=0):
        '''
        Performs Non-Negative Matrix Factorization (NMF) of a compressed
        sparse matrix using the sklearn.decomposition library.

        Parameters:
            csr (scipy.sparse.csr_matrix): the sparse matrix

            k (int): the number of principal components to use

            init (str): the method used to initialize the procedure. Defaults
                        to 'nndsvd' (Nonnegative Double Singular Value
                        Decomposition).

            max_iter (int): the maximum number of iterations to perform

            random_state (int):

            alpha (float):

        Returns:
            (array): the row by component aspect of the data. For example, if
                the original matrix had users as rows and jobs as 
                features/columns, was of size 1000 x 20, and was decomposed into
                2 components, the method would return a user-job aspect matrix
                of size 1000 by 2.

            (array): the component by feature aspect of the data. For example,
                if the original matrix had users as rows and jobs as 
                features/columns, was of size 1000 x 20, and was decomposed into
                2 components, the method would return a job aspect-job matrix
                of size 2 by 20.
        '''
        model = NMF(
            n_components=k,
            init=init,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha)
        W = model.fit_transform(csr)
        H = model.components_
        return W, H

    
    def tune_hyperparameters(
        self,
        window_id=1,
        users_as_rows=True,
        app_threshold=5,
        pct_test=0.2,
        subset_pct=0.5,
        num_iterations=10,
        max_nmf_iterations=500,
        k_range=(10, 20), 
        random_mask=True):
        '''
        '''
        # Initialize list of auc metrics
        aucs = []

        # Initialize list of best number of latent features at each iteration
        best_latent_features = []

        # Initialize best model
        best_model = None

        # Initialize best mean AUC
        best_mean_auc = -1

        # Initialize best alphas
        best_alphas = []

        # Initialize list of regularization parameters to try
        alphas = [0]

        # Initialize range of latent factor counts to tune
        min_num_components = k_range[0]
        max_num_components = k_range[1] + 1

        # Iterate over range of random seeds
        for seed in range(num_iterations):

            self.log.header(f"Iteration {seed + 1}")

            # Split data into training and test sets. 
            # Different random seeds will ensure different splits of data.
            train, test, masked, user_lookup, job_lookup = self.create_train_and_test_sets(
                window_id=window_id,
                users_as_rows=users_as_rows,
                app_threshold=app_threshold,
                pct_test = 0.2, 
                seed=seed,
                random_mask=random_mask,
                log_disabled=True)

            # Keep track of AUC metrics for each CV fold
            cv_auc = {}

            # Iterate over range of latent features
            for num_components in range(min_num_components, max_num_components):
                for alpha in alphas:

                    # Define model
                    # Note: Initialize estimates using non-negative double SVD, 
                    # which is better for sparseness
                    self.log.header(f"Running NMF with k = {num_components} "
                        f"and alpha = {alpha}")
                    user_vecs, item_vecs = self.perform_nmf(
                        train,
                        k=num_components, 
                        init="nndsvd", 
                        max_iter=max_nmf_iterations, 
                        random_state=0, 
                        alpha=alpha)

                    # Calculate mean AUC on test data
                    mean_auc, popular_auc = self.calc_mean_auc(
                        train=train, 
                        masked_apps=masked, 
                        predictions=[csr_matrix(user_vecs), csr_matrix(item_vecs)], 
                        test=test,
                        user_lookup=user_lookup,
                        job_lookup=job_lookup)

                    # Update best mean AUC and model
                    if mean_auc >= best_mean_auc:
                        best_mean_auc = mean_auc
                        best_model = (user_vecs, item_vecs)

                    # Log results
                    self.log.detail(f"Number of Components: {num_components}")
                    self.log.detail(f"Mean AUC: {mean_auc}")
                    self.log.detail(f"Popular AUC: {popular_auc}")

                    # Keep track of AUC score corresponding to each number 
                    # of latent features and regularization param
                    cv_auc[(num_components, alpha)] = (mean_auc, popular_auc)

            # Define the "best" number of latent features 
            # as the one with the highest AUC
            best_latent_feature = self.get_key(cv_auc, max(cv_auc.values()))[0]
            best_alpha = self.get_key(cv_auc, max(cv_auc.values()))[1]
            best_alphas.append(best_alpha)

            #plt.plot(cv_auc)
            best_latent_features.append(best_latent_feature)
            cv_auc_mean = np.mean(list(cv_auc.values()))
            aucs.append(cv_auc_mean)

        # Define best number of latent features as rounded mean of best latent features across random seeds
        overall_best_latent_feature = int(np.round(np.mean(best_latent_features)))
        overall_best_alpha = np.mean(best_alphas)
        self.log.header(f"Optimal number of latent features: {overall_best_latent_feature}")
        self.log.header(f"Optimal regularization parameter: {overall_best_alpha}")

        return overall_best_latent_feature, overall_best_alpha, best_model

       
    def train_data(
        self,
        num_components,
        alpha,
        window_id=1,
        users_as_rows=True,
        app_threshold=5,
        pct_test=0.2,
        subset_pct=0.5,
        num_iterations=10,
        max_nmf_iterations=500,
        random_mask=True):
        '''
        '''
        # Keep track of AUC metrics for each CV fold
        mean_aucs_for_folds = []
        popular_aucs_for_folds = []
        group_aucs_for_folds = []
        group_preds_for_folds = []
        group_actual_for_folds = []

        # Iterate over range of random seeds
        for seed in range(num_iterations):

            self.log.header(f"Fold {seed + 1}")

            # Split data into training and test sets. 
            # Different random seeds will ensure different splits of data.
            train, test, masked, user_lookup, job_lookup = self.create_train_and_test_sets(
                window_id=window_id,
                users_as_rows=users_as_rows,
                app_threshold=app_threshold,
                pct_test=0.2,
                subset_pct=subset_pct,
                seed=seed,
                random_mask=random_mask,
                log_disabled=True)

            # Define model/run NMF
            # Note: Initialize estimates using non-negative double SVD, 
            # which is better for sparseness
            self.log.detail(f"Running NMF with k = {num_components} "
                f"and alpha = {alpha}")
            user_vecs, item_vecs = self.perform_nmf(
                train,
                k=num_components, 
                init="nndsvd", 
                max_iter=max_nmf_iterations, 
                random_state=0, 
                alpha=alpha)

            # Calculate mean AUC on overall test data
            mean_auc, popular_auc = self.calc_mean_auc(
                train=train, 
                masked_apps=masked, 
                predictions=[csr_matrix(user_vecs), csr_matrix(item_vecs)], 
                test=test,
                user_lookup=user_lookup,
                job_lookup=job_lookup)
            
            # Calculate mean AUC of user groups specifically
            group_actual, group_preds, group_aucs = self.get_group_aucs(user_lookup, 
                                                          user_vecs, 
                                                          item_vecs, 
                                                          test)

            # Store results
            mean_aucs_for_folds.append(mean_auc)
            popular_aucs_for_folds.append(popular_auc)
            group_aucs_for_folds.append(group_aucs)
            group_preds_for_folds.append(group_preds)
            group_actual_for_folds.append(group_actual)

            # Log results
            self.log.detail(f"Number of Components: {num_components}")
            self.log.detail(f"Mean AUC: {mean_auc}")
            self.log.detail(f"Popular AUC: {popular_auc}")
            self.log.detail(f"Group AUCs:")
            for name, auc in group_aucs.items():
                self.log.detail(f"- {name}: {auc}")

        return mean_aucs_for_folds, popular_aucs_for_folds, group_aucs_for_folds, group_preds_for_folds, group_actual_for_folds


    ## EVALUATION ##
    def auc_score(self, predictions, targets):
        '''
        Outputs the area under the curve using sklearn's metrics. 
        
        Parameters:
            predictions (): your prediction output
            targets (): the actual target results you are comparing to
        
        Returns:
            (): the AUC (area under the Receiver Operating Characterisic curve)
        '''
        fpr, tpr, thresholds = metrics.roc_curve(targets, predictions)
        return metrics.auc(fpr, tpr)


    def avg_auc_score(self, predicted, actual):
        '''
        Calculates average AUC score without test data.
        '''
        rows, cols = predicted.shape
        scores = []

        for i in range(rows):
            pred_row = predicted[i,:].toarray().reshape((cols, 1))
            actual_row = actual[i,:].toarray().reshape((cols, 1))
            score = metrics.roc_auc_score(actual_row, pred_row)
            scores.append(score)

        return np.mean(scores)


    def calc_mean_auc(
        self, 
        train,
        test, 
        masked_apps, 
        predictions, 
        user_lookup,
        job_lookup):
        '''
        Calculates the mean AUC by user for any user that had their 
        user-item matrix altered. 
        
        Parameters: 
            train (scipy.sparse.csr_matrix): the training set, where a certain 
                                            percentage of the original user/item
                                            interactions are reset to zero 
                                            to hide them from the model 
            
            masked_apps (): the indices of the users where at least one 
                             user/item pair was masked
      
            predictions (): the matrix of predictions for each 
                            user/item pair as output from the implicit MF.
                            These should be stored in a list, with user vectors 
                            as item zero and item vectors as item one. 
            
            test (scipy.sparse.csr_matrix): the test set
           
        Returns:   
            (): The mean AUC (area under the Receiver Operator Characteristic 
                curve) of the test set only on user-item interactions there 
                were originally zero to test ranking ability in addition to the 
                most popular items as a benchmark.
        '''
        # Initialize empty list to store the AUC for each masked user 
        user_auc = []

        # Initialize empty list to store popular AUC scores
        popularity_auc = []

        # Get sum of item interactions to find most popular
        popular_items = np.array(test.sum(axis=0)).reshape(-1)

        # Retrieve predictions (i.e., sklearn's W and H)
        user_vecs, item_vecs = predictions

        # Iterate through each user that had an application masked
        for user_id in masked_apps["UserID"]:

            # Get the training set row
            user_idx = user_lookup[user_id]
            training_row = train[user_idx,:].toarray().reshape(-1)

            # Find where the interaction had not yet occurred
            zero_inds = np.where(training_row == 0)

            # Get the predicted values based on our user/item vectors
            user_vec = predictions[0][user_idx,:]
            pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)

            # Get only the items that were originally zero
            # Select all ratings from the MF prediction for this user 
            # that originally had no interaction
            actual = test[user_idx, :].toarray()[0, zero_inds].reshape(-1)

            # Select the binarized yes/no interaction pairs from the original 
            # full data that align with the same pairs in training

            # Get the item popularity for our chosen items
            pop = popular_items[zero_inds]

            # Calculate AUC for the given user and job application
            user_auc.append(self.auc_score(pred, actual)) 

            # Calculate AUC using most popular job application and score
            popularity_auc.append(self.auc_score(pop, actual))
        
        # Return the mean AUC rounded to three decimal places for both 
        # test and popularity benchmark
        return float('%.3f'%np.mean(user_auc)), float('%.3f'%np.mean(popularity_auc))  


    def get_key(self, dictionary, val):
        '''
        Retrieves the first key in a given dictionary that is
        associated with a given value.

        Parameters:
            dictionary (dict): the dictionary
            val (dynamic): the value

        Returns:
            (dynamic): the first key corresponding to the value, or None if
                       no keys are found
        '''
        for key, value in dictionary.items():
            if val == value:
                return key
    
        return None


    def get_group_aucs(self, user_lookup, user_vecs, item_vecs, test):
        '''
        '''
        # Create DataFrame to look up trained users' group affiliations
        data = {
            'Index': list(user_lookup.values()), 
            'UserID': list(user_lookup.keys())
        }
        data = pd.DataFrame(data)
        group_lookup = data.merge(self.users, on="UserID", how="inner")
        display(group_lookup)

        # Initialize function to calculate a single group's average AUC
        def calc_group_avg_auc(group_name, group_lookup):
            grp_indices = (group_lookup
                .query(f"Group == '{group_name}'")["Index"]
                .tolist())
            grp_actual = test[grp_indices, :]
            grp_pred = csr_matrix(user_vecs.dot(item_vecs))[grp_indices, :]
            grp_avg_auc = self.avg_auc_score(grp_pred, grp_actual)

            return grp_pred, grp_avg_auc, grp_actual

        # Store predictions and average AUCs for each group
        group_predictions = {}
        group_actual = {}
        group_aucs = {}
        group_names = self.users["Group"].unique().tolist()

        for name in group_names:
            grp_pred, grp_avg_auc, grp_actual = calc_group_avg_auc(name, group_lookup)
            group_predictions[name] = grp_pred
            group_aucs[name] = grp_avg_auc
            group_actual[name] = grp_actual

        return group_actual, group_predictions, group_aucs
        

    ## FAIRNESS ##
    def value_unfairness(
        self, 
        advantaged_true,
        advantaged_pred, 
        disadvantaged_true,
        disadvantaged_pred):
        '''
        Measures inconsistency in signed estimation error across
        the user types
        
        Value unfairness occurs when one class of user is consistently 
        given higher or lower predictions than their true preferences. If 
        the errors in prediction are evenly balanced between overestimation
        and underestimation or if both classes of users have the same 
        direction and magnitude of error, the value unfairness becomes 
        small. Value unfairness becomes large when predictions for one class
        are consistently overestimated and predictions for the other class 
        are consistently underestimated.
        '''
        # rows, cols = advantaged.shape
        # n = cols
        # ddiff = disadvantaged.mean(axis=0) - np.mean(disadvantaged)
        # adiff = advantaged.mean(axis=0) - np.mean(advantaged)
        # return 1/ n * np.sum(abs(ddiff - adiff))

        rows, cols = advantaged_true.shape
        n = cols
        ddiff = disadvantaged_pred.mean(axis=0) - disadvantaged_true.mean(axis=0)
        adiff = advantaged_pred.mean(axis=0) - advantaged_true.mean(axis=0)
        return (1 / n) * np.sum(abs(ddiff - adiff))



    def absolute_unfairness(
        self, 
        advantaged_true,
        advantaged_pred, 
        disadvantaged_true,
        disadvantaged_pred):
        '''
        Measures inconsistency in absolute estimation error
        across user types
        
        Absolute unfairness is unsigned, so it captures a single statistic 
        representing the quality of prediction for each user type. If one user 
        type has small reconstruction error and the other user type has large 
        reconstruction error, one type of user has the unfair advantage of good 
        recommendation, while the other user type has poor recommendation. 
        In contrast to value unfairness, absolute unfairness does not consider 
        the direction of error. 
        
        For example, if female students are given predictions 0.5 points
        below their true preferences and male students are given predictions 
        0.5 points above their true preferences, there is no absolute unfairness.
        '''
        rows, cols = advantaged_true.shape
        n = cols
        ddiff = abs(disadvantaged_pred.mean(axis=0) - disadvantaged_true.mean(axis=0))
        adiff = abs(advantaged_pred.mean(axis=0) - advantaged_true.mean(axis=0))
        return (1/ n) * np.sum(abs(ddiff - adiff))


    def underestimation_unfairness(
        self, 
        advantaged_true,
        advantaged_pred, 
        disadvantaged_true,
        disadvantaged_pred):
        '''
        Measures inconsistency in how much the predictions underestimate
        the true ratings.
        
        Underestimation unfairness is important in settings where missing 
        recommendations are more critical than extra recommendations. For 
        example, underestimation could lead to a top student not being
        recommended to explore a topic they would excel in.
        '''
        rows, cols = advantaged_true.shape
        n = cols

        ddiff = disadvantaged_true.mean(axis=0) - disadvantaged_pred.mean(axis=0)
        adiff = advantaged_true.mean(axis=0) - advantaged_pred.mean(axis=0)

        ddiff_clip = ddiff.clip(min=0)
        adiff_clip = adiff.clip(min=0)
        return (1 / n) * np.sum(abs(ddiff_clip - adiff_clip))


    def overestimation_unfairness(
        self, 
        advantaged_true,
        advantaged_pred, 
        disadvantaged_true,
        disadvantaged_pred):
        '''
        Measures inconsistency in how much the predictions overestimate 
        the true ratings.
        
        Overestimation unfairness may be important in settings where users 
        may be overwhelmed by recommendations, so providing too many 
        recommendations would be especially detrimental.
        
        For example, if users must invest large amounts of time to evaluate each 
        recommended item, overestimating essentially costs the user time. 
        Thus, uneven amounts of overestimation could cost one type of user 
        more time than the other.
        ''' 
        rows, cols = advantaged_true.shape
        n = cols

        ddiff = disadvantaged_pred.mean(axis=0) - disadvantaged_true.mean(axis=0)
        adiff = advantaged_pred.mean(axis=0) - advantaged_true.mean(axis=0)

        ddiff_clip = ddiff.clip(min=0)
        adiff_clip = adiff.clip(min=0)
        return (1 / n) * np.sum(abs(ddiff_clip - adiff_clip))


    def nonparity_unfairness(
        self,
        advantaged_true,
        advantaged_pred, 
        disadvantaged_true,
        disadvantaged_pred):
        '''
        Measure based on the regularization term introduced by Kamishima
        et al. Can be computed as the absolute difference between the 
        overall average ratings of disadvantaged users and 
        those of advantaged users.
        '''
        return abs(np.mean(disadvantaged_pred) - np.mean(advantaged_pred))


    def fairness_df(
        self, 
        group_true_and_preds,
        unfairness_func, 
        group_names):
        '''
        Creates a DataFrame of fairness metrics between groups.      
        '''
        metrics = np.zeros((len(group_names), len(group_names)))
        for i, val1 in enumerate(group_true_and_preds):

            for j, val2 in enumerate(group_true_and_preds):

                advantaged_true, advantaged_pred = val1
                disadvantaged_true, disadvantaged_pred = val2
                metrics[i,j] = unfairness_func(advantaged_true, 
                                                advantaged_pred, 
                                                disadvantaged_true, 
                                                disadvantaged_pred)
    
        data = pd.DataFrame(metrics, columns=group_names)
        data["group_names"] = group_names
        data = data.set_index("group_names")

        return data

