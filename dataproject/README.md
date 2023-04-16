# Data analysis project
My project is titled "Depreciation rates" and is about depreciation rates across different sectors. I might extend the analysis before the exam.

Structure:
- data
    - external: Mappings and other external inputs
    - processed: Processed data
    - raw: Raw data
- notebooks
    - depreciation_rates.ipynb: Results are generated here
- src
    - data
        - execute.py: calls extract.py and transform.py
        - extract.py: extracts data from DST's API and stores it in raw data folder
        - transform.py: transforms data and stores it in processed data folder
    - visualization
        - visualize.py: Code for plots


Dependencies:
Other than the standard Anaconda Python 3 installation, the file requires the PyDST package. This can be installed by running the following command in the terminal:
- pip install git+https://github.com/Kristianuruplarsen/pydst.git