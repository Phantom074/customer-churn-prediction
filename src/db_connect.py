"""
db_connect.py
=============
PostgreSQL connection helper using SQLAlchemy + psycopg2.
Author: Mukul (github.com/phantom074)
"""

import yaml
import pandas as pd
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_engine(config_path: str = "config.yaml"):
    """Create and return a SQLAlchemy engine from config.yaml."""
    cfg = load_config(config_path)["database"]
    url = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['name']}"
    engine = create_engine(url)
    logger.info(f"Connected to PostgreSQL: {cfg['host']}:{cfg['port']}/{cfg['name']}")
    return engine


def run_query(query: str, config_path: str = "config.yaml") -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    engine = get_engine(config_path)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df


def load_table(table_name: str, config_path: str = "config.yaml") -> pd.DataFrame:
    """Load an entire table into a DataFrame."""
    return run_query(f"SELECT * FROM {table_name}", config_path)


def write_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = "replace", config_path: str = "config.yaml"):
    """Write a DataFrame to PostgreSQL."""
    engine = get_engine(config_path)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    logger.info(f"Written {len(df)} rows to table '{table_name}'")


def test_connection(config_path: str = "config.yaml") -> bool:
    """Test if the database connection works."""
    try:
        engine = get_engine(config_path)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful!")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False
