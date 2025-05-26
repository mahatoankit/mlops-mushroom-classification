import json
import redis
import pandas as pd
import numpy as np
from io import StringIO
import logging

logger = logging.getLogger(__name__)

class RedisClient:
    """
    Client for interacting with Redis to store and retrieve data for the mushroom classification pipeline.
    """
    
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        """
        Initialize Redis client connection.
        
        Args:
            host (str): Redis server host
            port (int): Redis server port
            db (int): Redis database to use
            password (str, optional): Redis password for authentication
        """
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False  # Keep as bytes for binary data
            )
            self.client.ping()  # Test connection
            logger.info(f"Successfully connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def store_dataframe(self, key, df, expiry=None):
        """
        Store pandas DataFrame in Redis.
        
        Args:
            key (str): Redis key to store the data under
            df (pd.DataFrame): DataFrame to store
            expiry (int, optional): Time in seconds until data expires
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert DataFrame to JSON string
            df_json = df.to_json(orient='split')
            
            # Store in Redis
            if expiry:
                self.client.setex(key, expiry, df_json)
            else:
                self.client.set(key, df_json)
                
            logger.info(f"Successfully stored DataFrame with key '{key}', shape: {df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error storing DataFrame in Redis: {e}")
            return False

    def get_dataframe(self, key):
        """
        Retrieve DataFrame from Redis.
        
        Args:
            key (str): Redis key to retrieve
            
        Returns:
            pd.DataFrame: Retrieved DataFrame or None if not found
        """
        try:
            df_json = self.client.get(key)
            if df_json is None:
                logger.warning(f"No data found in Redis with key '{key}'")
                return None
                
            # Convert JSON string back to DataFrame
            df = pd.read_json(StringIO(df_json.decode('utf-8')), orient='split')
            logger.info(f"Successfully retrieved DataFrame with key '{key}', shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error retrieving DataFrame from Redis: {e}")
            return None
    
    def delete_key(self, key):
        """
        Delete a key from Redis.
        
        Args:
            key (str): Redis key to delete
            
        Returns:
            bool: True if successful
        """
        try:
            result = self.client.delete(key)
            if result:
                logger.info(f"Successfully deleted key '{key}' from Redis")
                return True
            else:
                logger.warning(f"Key '{key}' not found in Redis")
                return False
        except Exception as e:
            logger.error(f"Error deleting key from Redis: {e}")
            return False
            
    def store_model_metadata(self, model_id, metadata):
        """
        Store model metadata in Redis.
        
        Args:
            model_id (str): Unique identifier for the model
            metadata (dict): Model metadata to store
            
        Returns:
            bool: True if successful
        """
        try:
            key = f"model:{model_id}:metadata"
            self.client.set(key, json.dumps(metadata))
            logger.info(f"Stored metadata for model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing model metadata: {e}")
            return False
            
    def get_model_metadata(self, model_id):
        """
        Retrieve model metadata from Redis.
        
        Args:
            model_id (str): Unique identifier for the model
            
        Returns:
            dict: Model metadata or None if not found
        """
        try:
            key = f"model:{model_id}:metadata"
            data = self.client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Error retrieving model metadata: {e}")
            return None