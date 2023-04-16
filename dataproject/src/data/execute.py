from src.data.extract import extract
from src.data.transform import transform


def execute():
    extract()
    transform()
    

if __name__ == '__main__':
    execute()