# Data analysis project
My project is titled "Markups" and is about depreciation rates across different sectors. 

Structure:
- markups.ipynb: Results are generated here
- data
    - external: Mappings and other external inputs
    - processed: Processed data
    - raw: Raw data
- src
    - data
        - extract.py: extracts data from DST's API and stores it in raw data folder
        - transform.py: transforms data and stores it in processed data folder


Dependencies:
Other than the standard Anaconda Python 3 installation, the file requires the PyDST package. This can be installed by running the following command in the terminal:
- pip install git+https://github.com/Kristianuruplarsen/pydst.git