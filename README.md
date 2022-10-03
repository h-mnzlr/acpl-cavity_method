# Running the code:
1. Clone the repository `git clone https://github.com/h-mnzlr/acpl-cavity_method.git`
2. Make sure you run the correct python version `python --version`. Output should be `Python 3.10.6`.
3. Create a virtual environment at e.g. `env/`: `python -m venv env`
4. Activate the environment `source env/bin/activate`
5. Install all requirement: `pip install -r requirements.txt`
6. Dynamically link the local code packages into the environment `pip install -e .`
7. Spin up the Jupyter server using `jupyter notebook`
8. Run the code from the notebooks.

# Repo structure
##### `notebooks/`
Contains the notebooks that implement the exercises. Notebooks are called `.sync.ipynb` due to workflow reasons (`jupyter_ascending` Jupyter server plugin).

##### `code/`
Contains all the packages dynamically linked in the environment. Contains both the provided libraries and modules with self-implemented
helper functions to use in the notebooks.
