from typing import Optional

from pymongo import MongoClient

from config.settings import settings


_mongo_client: Optional[MongoClient] = None
_database = None


def connect_to_mongo() -> MongoClient:
    global _mongo_client, _database

    if _mongo_client is None:
        _mongo_client = MongoClient(settings.mongo_uri)
        _database = _mongo_client[settings.mongo_db_name]

    return _mongo_client


def get_database():
    global _database

    if _database is None:
        connect_to_mongo()

    return _database


def close_mongo_connection() -> None:
    global _mongo_client

    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
