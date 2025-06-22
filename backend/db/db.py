from pymongo import MongoClient
import urllib.parse
import os
from utils.utils import log_message

DB_USERNAME = os.environ.get("DB_USERNAME")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
parsed_username = urllib.parse.quote_plus(DB_USERNAME)
parsed_password = urllib.parse.quote_plus(DB_PASSWORD)
DB_NAME = os.environ.get("DB_NAME")
DB_STR = os.environ.get("DB_STR")
DB_CLUSTER = os.environ.get("DB_CLUSTER")
MONGODB_URL = f"mongodb+srv://{parsed_username}:{parsed_password}@{DB_CLUSTER}.{DB_STR}.mongodb.net/?retryWrites=true&w=majority&appName={DB_CLUSTER}"

client = MongoClient(MONGODB_URL)


class DatabaseInstantiator:
    def __init__(self):
        self.db_name = DB_NAME

    def does_database_exist(self) -> bool:
        database_names = client.list_database_names()
        if self.db_name not in database_names:
            return False
        return True

    def return_database_object(self):
        if self.does_database_exist() is True:
            log_message("warning", f"database {self.db_name} already exists")
            pass
        return client.get_database(self.db_name)

    def get_table_collection(self, name):
        collection = self.return_database_object().get_collection(name)
        log_message("info", f"getting table collection for {name}")
        return collection
