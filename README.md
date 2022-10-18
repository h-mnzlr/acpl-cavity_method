# Running the code:
1. Clone the repository `git clone https://github.com/h-mnzlr/acpl-cavity_method.git`
2. Make sure you run the correct python version `python --version`. Output should be `Python 3.10.8`.
3. Create a virtual environment at e.g. `env/`: `python -m venv env`
4. Activate the environment `source env/bin/activate`
5. Dynamically link the local code packages into the environment `pip install -e .`
6. Install all requirement: `pip install -r requirements.txt`
7. Spin up the Jupyter server using `jupyter notebook`
8. Run the code from the notebooks.


# Repo structure
##### `notebooks/`
Contains the notebooks that implement the exercises. Notebooks are called `.sync.ipynb` due to workflow reasons (`jupyter_ascending` Jupyter server plugin).

Exercise 1: `ising_model.sync.ipynb`
Exercise 2: `1_random_matrices.ipynb` and `2_random_matrices.ipynb`
Exercise 3: `anderson_model.ipynb`

##### `code/`
Contains all the packages dynamically linked in the environment. Contains modules with self-implemented helper functions to use in the notebooks.

# Note to Grader:
Feel free to browse through the notebooks on github directly.
Some of the simulation are coded so that they can run in parallel on a large cluster.
If you try to run them locally, runtime might be too large to actually reproduce the results.
