"""
    database / processors.py
    ------------------------
    Data-Processor retrieves the loaded data from pd.DataFrame and checks the 
    structure of data and validate it based on the expected appearance of
    datasets.
"""
from __future__ import annotations

import orjson
import numpy as np
import pandas as pd
import time

from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from numpy.typing import DTypeLike

from logs.logger import Logger, Level



class DataProcessor:
    """
    Handles structured transformations of input tabular data such as 
    pd (Pandas).DataFrame into a more consistent NumPy arrays
    """
    SCHEMA = {
        "dvec"      : {"dtype": np.float32, "stacked": True, "dim": 3},
        "rx_type"   : {"dtype": np.uint8,   "stacked": False},
        "link_state": {"dtype": np.uint8,   "stacked": False},
        "los_pl"    : {"dtype": np.float32, "stacked": False},
        "los_ang"   : {"dtype": np.float32, "stacked": True, "dim": 4},
        "los_dly"   : {"dtype": np.float32, "stacked": False},
        "nlos_pl"   : {"dtype": np.float32, "stacked": True, "dim": 20},
        "nlos_ang"  : {"dtype": np.float32, "stacked": True, "dim": (20, 4)},
        "nlos_dly"  : {"dtype": np.float32, "stacked": True, "dim": 20},
    }

    
    def __init__(self, level: Level = Level.INFO):
        """
        Initialize Data-Processor Instance
        """
        self._dtype_cache: Dict[str, np.ndarray] = {}
        
        self.logger = Logger("Data-Processor", level)
        self.logger.info("DataProcessor initialized")
        self.logger.debug(f"Schema loaded with {len(self.SCHEMA)} columns")


    def process(self, chunk: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Process a pandas DataFrame chunk into a dictionary of numpy arrays.
        
        Args:
            chunk: Pandas DataFrame to process
            
        Returns:
            Dictionary of processed numpy arrays
        """
        start_time = time.time()
        if chunk.empty:
            self.logger.warning("Received empty DataFrame chunk")
            return {}
        
        self.logger.debug(
            f"Processing chunk with {len(chunk)} rows, {len(chunk.columns)} columns"
        )
        
        processed: Dict[str, np.ndarray] = {}
        failed_columns: List[str] = []
        success_count = 0
        
        for column, spec in self.SCHEMA.items():
            col_start = time.time()

            if column not in chunk.columns:
                self.logger.debug(f"Column '{column}' not found in chunk, skipping")
                continue

            dtype, stacked = spec["dtype"], spec.get("stacked", False)
            values = chunk[column].to_numpy(copy=False)

            if values.size == 0:
                self.logger.debug(f"Column '{column}' is empty")
                processed[column] = np.empty((0,), dtype=dtype)
                success_count += 1
                continue

            try:
                if not stacked:
                    processed[column] = self._convert_simple_column(values, dtype, column)
                    success_count += 1
                    self.logger.debug(
                        f"Column '{column}' converted in "
                        f"{time.time() - col_start:.4f}s"
                    )
        
                else:
                    processed[column] = self._convert_stacked_column(values, dtype, column)
                    success_count += 1
                    self.logger.debug(
                        f"Stacked column '{column}' converted in "
                        f"{time.time() - col_start:.4f}s"
                    )
            
            except Exception as e:
                self.logger.warning(f"Failed to process column '{column}': {e}, using dtype=object")
                processed[column] = np.asarray(values, dtype=object)
                failed_columns.append(column)
        
        # Log processing statistics
        elapsed = time.time() - start_time
        if failed_columns:
            self.logger.warning(
                f"Processed {success_count} columns successfully, "
                f"{len(failed_columns)} with fallback dtype: {failed_columns}"
            )
        
        missing = set(self.SCHEMA.keys()) - set(processed.keys())
        if missing:
            self.logger.warning(f"Missing {len(missing)} column(s) in processed output: {missing}")
        
        self.logger.debug(
            f"Processed {len(processed)} columns from chunk with {len(chunk)} rows "
            f"in {elapsed:.4f}s"
        )
        return processed
    

    def _convert_simple_column(self,
        values: np.ndarray, dtype: DTypeLike, column: str
    ) -> np.ndarray:
        """
        Convert a simple (non-stacked) column to the target dtype.
        
        Args:
            values: Input numpy array
            dtype: Target dtype
            column: Column name for logging
            
        Returns:
            Converted numpy array
            
        Raises:
            ValueError: If conversion fails
        """
        try:
            result = values.astype(dtype, copy=False)
            self.logger.debug(f"Column '{column}' astype successful: {dtype}")
            return result
        
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Column '{column}' contains non-castable values; using dtype=object")
            self.logger.debug(f"Conversion error for '{column}': {e}")
            return np.asarray(values, dtype=object)


    def _convert_stacked_column(self,
        values: np.ndarray, dtype: DTypeLike, column: str
    ) -> np.ndarray:
        """
        Convert a stacked column with nested data structures.
        
        Args:
            values: Input numpy array
            dtype: Target dtype
            column: Column name for logging
            
        Returns:
            Converted numpy array
        """
        sample = self._first_valid(values)

        if sample is None:
            self.logger.warning(f"Column '{column}' has no valid values, using object dtype")
            return np.asarray(values, dtype=object)
        
        if isinstance(sample, str):
            self.logger.debug(f"Column '{column}' contains JSON strings, decoding")
            return self._decode_json_array(values, dtype, column)
        
        try:
            converted = np.asarray(values.tolist(), dtype=dtype)
            self.logger.debug(f"Successfully converted stacked column '{column}'")
            return converted
        
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Column '{column}' contains ragged arrays; fallback to object: {e}")
            return np.asarray(values, dtype=object)


    def _decode_json_array(self,
        values: np.ndarray, dtype: DTypeLike, column: str
    ) -> np.ndarray:
        """
        Decode JSON arrays from string values.
        
        Args:
            values: Array of JSON strings
            dtype: Target dtype
            column: Column name for logging
            
        Returns:
            Decoded numpy array
        """
        try:
            decoded: List[Any] = []
            null_count = 0
            
            for v in values:

                if v is None:
                    decoded.append(None)
                    null_count += 1
                
                else:
                    decoded.append(orjson.loads(v))
            
            if null_count > 0:
                self.logger.warning(f"Column '{column}' contains {null_count} null values")
            
            return np.asarray(decoded, dtype=dtype)
        
        except Exception as e:
            self.logger.warning(f"Column '{column}' contains malformed JSON; coercing to dtype=object: {e}")
            return np.array([self._safe_parse(v) for v in values], dtype=object)
    

    def _decode_json_array(self,
        values: np.ndarray, dtype: DTypeLike, column: str
    ) -> np.ndarray:
        """
        Decode JSON arrays from string values with performance optimizations.
        """
        try:
            # Pre-allocate list for avoid copy performance 
            decoded: List[Any] = [None] * len(values)
            null_count: int = 0
            loads = orjson.loads
            
            for i, v in enumerate(values):
                
                if v is None:
                    null_count += 1
                    decoded[i] = None
                
                else:
                    decoded[i] = loads(v)
            
            if null_count > 0:
                self.logger.warning(f"Column '{column}' contains {null_count} null values")
            
            self.logger.debug(f"Decoded {len(values)} JSON strings for column '{column}'")
            return np.asarray(decoded, dtype=dtype)
        
        except Exception as e:
            self.logger.warning(f"Column '{column}' contains malformed JSON; coercing to dtype=object: {e}")
            return np.array([self._safe_parse(v) for v in values], dtype=object)
    

    @staticmethod
    def _first_valid(values: np.ndarray) -> Optional[Any]:
        """
        Retrieves the first non-None value from passed array.
        
        Args:
            values: Array to search
            
        Returns:
            First non-None value, or None if all are None
        """
        for value in values:
        
            if value is not None:
                return value
        
        return None
    
    
    @staticmethod
    def _safe_parse(value: Any) -> Any:
        """
        Safely parse a value, trying JSON if it's a string.
        
        Args:
            value: Value to parse
            
        Returns:
            Parsed value or original if parsing fails
        """
        if isinstance(value, str):
            try:
                return orjson.loads(value)
            
            except Exception:
                return value
        
        return value


    def concatenate(self, results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Concatenate multiple processed results into a single dictionary.
        
        Args:
            results: List of processed data dictionaries
            
        Returns:
            Concatenated dictionary of numpy arrays
        """
        start_time = time.time()
        
        if not results:
            self.logger.warning("No results to concatenate, returning empty schema")
            
            return {
                key: np.empty((0,), dtype=spec["dtype"]) \
                    for key, spec in self.SCHEMA.items()
            }
        
        self.logger.info(f"Concatenating {len(results)} result chunks")
        
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        self.logger.debug(f"Found {len(all_keys)} unique keys across all chunks")
        
        output: Dict[str, np.ndarray] = {}
        failed_keys: List[str] = []
        concatenated_count = 0
        
        for key in all_keys:
            arrays: List[np.ndarray] = []
            valid_count = 0
            
            for result_idx, result in enumerate(results):
                
                if key in result:
                    arr = result[key]
                    valid_count += 1
                
                    if arr.ndim == 0:
                        arr = np.expand_dims(arr, axis=0)
                
                    arrays.append(arr)
            
            if not arrays:
                self.logger.debug(f"No arrays found for key '{key}', skipping")
                continue

            try:
                output[key] = np.concatenate(arrays, axis=0)
                concatenated_count += 1
                self.logger.debug(f"Concatenated {valid_count} arrays for '{key}'")
            
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Concatenation failed for '{key}': {e}, using object dtype")
                arrays = [np.asarray(array, dtype=object) for array in arrays]
                
                try:
                    output[key] = np.concatenate(arrays, axis=0)
                    concatenated_count += 1
                    self.logger.debug(f"Concatenated {valid_count} arrays for '{key}' (object dtype)")
                
                except Exception as e2:
                    self.logger.error(f"Fallback concatenation also failed for '{key}': {e2}")
                    failed_keys.append(key)
        
        if failed_keys:
            self.logger.warning(f"Failed to concatenate {len(failed_keys)} key(s): {failed_keys}")
        
        if output:
            lengths = {k: len(v) for k, v in output.items() if k in all_keys}
            if len(set(lengths.values())) > 1:
                self.logger.warning(f"Inconsistent lengths after concatenation: {lengths}")
        
        elapsed = time.time() - start_time
        row_count = len(next(iter(output.values()))) if output else 0
        
        self.logger.info(
            f"✅ Concatenated {concatenated_count} columns, "
            f"{row_count:,} rows in {elapsed:.4f}s"
        )
        
        return output
