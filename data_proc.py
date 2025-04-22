"""
Data Processing Utilities
========================

Comprehensive tools for handling various data formats and operations.
Includes functions for CSV, JSON, and database operations with robust
error handling and data validation.

Key Features:
    - CSV file operations with pandas
    - JSON processing
    - SQLite database interactions
    - Data validation and cleaning
"""

import pandas as pd
import sqlite3
import json
from typing import List, Dict 

class DataProcessor:
    """
    A class for handling various data processing operations.
    
    Attributes:
        None
        
    Methods:
        read_csv: Reads CSV files into pandas DataFrames
        write_csv: Writes DataFrames to CSV files
        process_json: Processes JSON data with validation
        clean_data: Performs basic data cleaning operations
    """
    
    @staticmethod
    def read_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame with extended functionality.
        
        Parameters:
            filepath (str): Path to the CSV file
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            pd.DataFrame: Processed DataFrame
            
        Example:
            >>> dp = DataProcessor()
            >>> df = dp.read_csv("data.csv", encoding='utf-8')
            >>> print(df.head())
        """
        return pd.read_csv(filepath, **kwargs)


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute SQL query and return results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()

    def create_table(self, table_name: str, columns: dict):
        """Create a table with specified columns."""
        columns_str = ", ".join([f"{col} {type}" for col, type in columns.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
        self.execute_query(query)

    def insert_data(self, table_name: str, data: list, columns: list):
        """Insert data into a specified table."""
        placeholders = ", ".join(["?"] * len(columns))
        columns_str = ", ".join(columns)
        query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(query, data)
            conn.commit()


