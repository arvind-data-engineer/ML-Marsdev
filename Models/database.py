import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url

logger = logging.getLogger(__name__)


def load_project_env(project_root):
    load_dotenv(project_root / ".env")


def get_env(name, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def require_env(name):
    value = get_env(name)
    if value is None:
        raise ValueError(f"Missing database environment variable: {name}")
    return value


def build_database_url():
    database_url = get_env("DATABASE_URL")
    if database_url:
        return database_url

    driver = get_env("DB_DRIVER", "postgresql+psycopg2")
    host = require_env("DB_HOST")
    database = require_env("DB_NAME")
    username = require_env("DB_USER")
    password = require_env("DB_PASSWORD")
    port = get_env("DB_PORT")

    query = {}
    sslmode = get_env("DB_SSLMODE")
    if sslmode:
        query["sslmode"] = sslmode

    return URL.create(
        drivername=driver,
        username=username,
        password=password,
        host=host,
        port=int(port) if port else None,
        database=database,
        query=query,
    )


def safe_database_label(database_url):
    try:
        url = make_url(str(database_url))
        return url.render_as_string(hide_password=True)
    except Exception:
        return "configured database"


def create_database_engine():
    database_url = build_database_url()
    logger.info("Connecting to %s.", safe_database_label(database_url))
    return create_engine(database_url, pool_pre_ping=True)
