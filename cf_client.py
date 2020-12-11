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

        # Load users
        self.users = pd.read_csv("data/users.tsv", sep="\t")
        display((HTML("<h4>users.tsv</h4>")))
        display(HTML("<p>Holds all users and their metadata</p>"))
        display(self.users.head(2))

        # Load apps
        self.apps = pd.read_csv("data/apps.tsv", sep="\t")
        display((HTML("<h4>apps.tsv</h4>")))
        display(HTML("<p>Holds the applications users submitted</p>"))
        display(self.apps.head(2))

        # Load jobs assigned to first window
        self.jobs1 = pd.read_csv("data/jobs1.tsv", sep="\t")
        display((HTML("<h4>jobs1.tsv</h4>")))
        display(HTML("<p>Holds the jobs available on CareerBuilder.com during a 13-day window</p>"))
        display(self.jobs1.head(2))

        # Load user history
        self.user_history = pd.read_csv("data/user_history.tsv", sep="\t")
        display((HTML("<h4>user_history.tsv</h4>")))
        display(HTML("<p>Holds users' past job title(s)</p>"))
        display(self.user_history.head(2))

        # Load window dates
        self.window_dates = pd.read_csv("data/window_dates.tsv", sep="\t")
        display((HTML("<h4>window_dates.tsv</h4>")))
        display(HTML("<p>Holds the application window dates</p>"))
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
        plt.spy(csr, markersize = 0.1, origin="lower")


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


    ## TRAINING AND TESTING ##
    def create_csr_matrix(self, app_ids, user_lookup, job_lookup, users_as_rows):
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

        display(HTML(f"<p>- Data Length: {len(data)}</p>"))

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
        pct_test=0.2, 
        seed=None,
        random_mask=True):
        '''
        Creates train and test CSR matrices. The matrices are identical
        except that the train matrix has masked (i.e., 0) values for test user
        applications submitted during the testing portion of the
        application window.

        Parameters:
            window_id (int): the window id

        Returns:
            (scipy.sparse.csr_matrix, scipy.sparse.csr_matrix): a tuple of
                CSR matrices.  The first element in the tuple is the training
                set and the second element is the testing set.
        '''
        display(HTML(f"<h5>Preparing data for application Window {window_id} " +
            "train and test sets...</h5>"))

        # Restrict applications to users meeting app submission threshold
        display(HTML(f"<h5>Restricting data to applications from users who submitted " + 
            f"{app_threshold} or more applications in the given window...</h5>"))
        app_ids = self.filter_applications(window_id, app_threshold)

        # Create id-index lookup for the qualified users
        user_ids = app_ids["UserID"].sort_values().unique().tolist()
        user_lookup = {val: idx for idx, val in enumerate(user_ids)}
        display(HTML(f"<p>- Users successfully filtered. {len(user_ids)} users submitted " +
            f"{app_threshold} or more apps in the window.</p>"))

        # Create id-index lookup for jobs in current window
        job_ids = app_ids["JobID"].sort_values().unique().tolist()
        job_lookup = {val: idx for idx, val in enumerate(job_ids)}
        display(HTML(f"<p>- Jobs successfully filtered. The qualified users submitted apps " +
            f"to {len(job_ids)} different jobs in the window.</p>"))
        display(HTML("<p>- Filtering complete.</p>"))

        # Mask a percentage of applications for testing
        display(HTML("<h5>Masking applications to use as test cells...</h5>"))
        masked_ids = self.mask_applications(app_ids, pct_test, seed, random_mask)
        display(HTML(f"<p>- {len(masked_ids)} app ids successfully masked with zeroes.</p>"))

        # Finalize application ids used for training and test sets
        display(HTML("<h5>Finalizing application ids to use in train and test sets...</h5>"))
        train_app_ids = (app_ids
            .merge(masked_ids, on=["UserID", "JobID"], how="outer", indicator=True)
            .query("_merge == 'left_only'")[["UserID", "JobID"]])
        test_app_ids = app_ids
        display(HTML("<h5>Data preparation complete.</h5>"))

        # Create training and testing matrices
        display(HTML("<h5>Creating train and test sets...</h5>"))
        display(HTML("<p>- Creating CSR train matrix...</p>"))
        train_matrix = self.create_csr_matrix(
            train_app_ids, 
            user_lookup, 
            job_lookup, 
            users_as_rows)

        display(HTML("<p>- Building CSR test matrix...</p>"))
        test_matrix = self.create_csr_matrix(
            test_app_ids,
            user_lookup,
            job_lookup,
            users_as_rows)

        display(HTML(f"<p>- Training and testing sets successfully created.</p>"))

        return (train_matrix, test_matrix)


    def filter_applications(self, window_id, app_threshold):
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
        app_counts = (self.apps
            .query("WindowID == @window_id")
            .groupby("UserID")
            .size()
            .to_frame()
            .rename(columns={0:"count"})
            .query("count >= @app_threshold"))

        # Display app counts for improved troubleshooting 
        app_counts_for_display = (app_counts["count"]
            .sort_values(ascending=False)
            .to_frame())

        display(app_counts_for_display)

        # Return app ids belonging to qualified users meeting threshold
        qualified_users = app_counts.index.tolist()
        app_ids = self.apps.query("UserID in @qualified_users")[["UserID", "JobID"]]

        return app_ids

    
    def mask_applications(self, app_ids, pct_test, seed, random_mask):
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

            display(HTML("<p>- Randomly masking applications in dataset...</p>"))

            # Initialize random seed
            random.seed(seed)
            display(HTML(f"<p>- Initializing random generator with seed {seed}...</p>"))

            # Initialize number of samples and population to draw from
            num_samples = int(np.ceil(pct_test * len(app_ids)))
            population = app_ids.index.tolist()
            display(HTML(f"<p>- Reading configuration setting...{pct_test * 100} " +
                "percent of apps should be used for testing out of the " +
                f"{len(population)} total available...<p>"))

            # Sample from population of indices
            samples = random.sample(population, num_samples)
            display(HTML(f"<p>- {len(samples)} app ids randomly chosen for masking...</p>"))

            # Mask apps corresponding to sampled/selected index values
            masked_ids = app_ids.loc[app_ids.index.isin(samples)]

        else:
            display(HTML("<p>- Masking apps made by test users in test window...</p>"))

            # Determine date range for testing window
            window_start = (self.window_dates
                .query("Window == @window_id")["Test Start"][0])
            window_end = (self.window_dates
                .query("Window == @window_id")["Test End"][0])
            display(HTML(f"<p>- The test window is {window_start} to {window_end}</p>"))

            # Filter apps to only include those in testing window
            masked_ids_query = (
                "WindowID == @window_id & " + 
                "Split == 'Test' & " + 
                "ApplicationDate >= @window_start & " + 
                "ApplicationDate <= @window_end")
            masked_ids = self.apps.query(masked_ids_query)[["UserID", "JobID"]]

        return masked_ids


    def perform_nmf(self, csr, k, init="nndsvd"):
        '''
        Performs Non-Negative Matrix Factorization (NMF) of a compressed
        sparse matrix using the sklearn.decomposition library.

        Parameters:
            csr (scipy.sparse.csr_matrix): the sparse matrix
            k (int): the number of principal components to use
            init (str): the method used to initialize the procedure. Defaults
                        to 'nndsvd' (Nonnegative Double Singular Value
                        Decomposition).

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
        model = NMF(n_components=k, init=init)
        W = model.fit_transform(csr)
        H = model.components_
        return W, H

    
