"""
    data / loader.py
    ----------------
    Loader object implementation that includes the structure of 
    loading the datasets for training the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import multiprocessing as mp
import time # --------------  # Perhaps using Timer instead

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed, Future
)
from database.processors import DataProcessor
from database.handlers import HandlerFactory, FileHandler

from logs.logger import Logger, Level



class DataLoader:
    """
    High-level loader for the datasets used for training the UAV model.
    This model includes manages the supported file formats to load the
    data into the program applying the appropriate file-handler. 
    Thereafter, data-processing is managing all the transformations 
    from the loaded data from pd.DataFrame into dictionary of NumPy 
    arrays for simplistic manipulation.
    """
    REQUIRED_COLUMNS = [
        'dvec', 'rx_type', 'link_state', 'los_pl',
        'los_ang', 'los_dly', 'nlos_pl', 'nlos_ang', 'nlos_dly'
    ]

    def __init__(self,
        n_workers: Optional[int]=None, chunk_size: int=10_000, 
        prefer_processes: bool=False, level: Level = Level.INFO
    ):
        """
            Initialize Data-Loader instance
        """
        self.directory = Path(__file__).parent / "datasets"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.n_workers = n_workers or mp.cpu_count()
        self.chunk_size = max(100, chunk_size)
        self.prefer_processes = prefer_processes

        self.processor = DataProcessor()
        self.logger = Logger("Data-Loader", level)
        self.logger.info(
            f"DataLoader initialized with {self.n_workers} workers, "
            f"chunk_size={self.chunk_size}, "
            f"mode={'process' if prefer_processes else 'thread'}"
        )
        
        self.logger.debug(f"Data directory: {self.directory.absolute()}")
    

    def load(self, filepaths: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Load data from one or more files and process them into numpy arrays.
        
        Args:
            filepaths: Single file path or list of file paths
            
        Returns:
            Dictionary of processed numpy arrays
            
        Raises:
            RuntimeError: If no data could be processed successfully
            FileNotFoundError: If a file doesn't exist
            ValueError: If file format is unsupported
        """
        if isinstance(filepaths, (str, Path)): 
            filepaths = [filepaths]
        
        self.logger.info(
            f"📂 Loading {len(filepaths)} file(s) with {self.n_workers} worker(s)..."
        )
        self.logger.debug(f"Files to load: {filepaths}")

        start_time = time.time() # Start the timer
        executor_class = ProcessPoolExecutor if self.prefer_processes \
            else ThreadPoolExecutor
    
        chunks: List[Dict[str, np.ndarray]] = []
        failed_files: List[str] = []
        total_chunks = 0
        total_rows_processed = 0
        
        for filepath in filepaths:
            path = Path(filepath) if Path(filepath).is_absolute() \
                else self.directory / filepath
            
            # perhaps some debugging message if later needed 
            if not path.exists():
                self.logger.error(f"File not found: ``{path}``")
                failed_files.append(str(path))
                continue

            try:
                file_start = time.time()
                handler = HandlerFactory.get_handler(path)
                self.logger.debug(
                    f"Processing {path} with {handler.__class__.__name__}"
                )

                with executor_class(max_workers=self.n_workers) as executor:
                    
                    futures: List[Future] = []
                    chunk_count = 0
                    file_rows = 0
                    
                    for chunk in handler.load_chunks(path, self.chunk_size):
                        chunk_count += 1
                        total_chunks += 1
                        file_rows += len(chunk)
                        self.logger.debug(
                            f"Chunk #{chunk_count} from ``{path.name}``: "
                            f"``{len(chunk)}`` rows"
                        )
                        
                        missing = [
                            column for column in self.REQUIRED_COLUMNS 
                            if column not in chunk.columns
                        ]
                        
                        if missing:
                            self.logger.warning(
                                f"Missing columns in ``{path.name}`` "
                                f"chunk #{chunk_count}: ``{missing}``"
                            )
                            continue

                        future = executor.submit(self.processor.process, chunk)
                        futures.append(future)
                    
                    self.logger.debug(
                        f"Submitted {len(futures)} chunks from {path} for processing"
                    )
                    
                    # Collect results with timeout
                    completed = 0
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=40)  # 40 seconds timeout
                            completed += 1

                            if result:
                                chunks.append(result)
                                total_rows_processed += len(next(iter(result.values())))
                                self.logger.debug(
                                    f"Processed chunk {completed}/{len(futures)} "
                                    f"from ``{path.name}``"
                                )
                            
                            else:
                                self.logger.warning(
                                    f"Empty result from chunk in ``{path}``"
                                )
                        
                        except TimeoutError:
                            self.logger.error(f"Chunk processing timed out for {path}")
                        
                        except Exception as e:
                            self.logger.error(f"Chunk processing failed for {path}: {e}")
                            self.logger.debug(f"Exception details:", exc_info=True)

                elapsed = time.time() - file_start
                self.logger.info(
                    f"✅ Processed {path.name}: {file_rows} rows in {elapsed:.2f}s"
                )
                            
            except FileNotFoundError as e:
                self.logger.error(f"File error processing {path}: {e}")
                failed_files.append(str(path))
                continue
                
            except ValueError as e:
                self.logger.error(f"Value error processing {path}: {e}")
                failed_files.append(str(path))
                continue
                
            except Exception as e:
                self.logger.error(f"Unexpected error processing {path}: {e}")
                failed_files.append(str(path))
                continue
        
        self.logger.debug(f"Processing summary:")
        self.logger.debug("--------------------------------------------------")
        self.logger.debug(f"  - Total chunks submitted: {total_chunks}")
        self.logger.debug(f"  - Total rows processed: {total_rows_processed:,}")
        self.logger.debug(f"  - Successful chunks: {len(chunks)}")

        if failed_files:
            self.logger.warning(
                f"Failed to process {len(failed_files)} file(s): {failed_files}"
            )
        
        if not chunks:
            msg = f"No data successfully processed from {len(filepaths)} file(s)"
            self.logger.error(msg)
            raise RuntimeError(msg)
        
        self.logger.debug(f"Concatenating {len(chunks)} processed chunks")
        try:
            processed = self.processor.concatenate(chunks)

        except Exception as e:
            msg = f"Failed to concatenate processed chunks: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e
        
        if not self.validate_data(processed):
            msg = "Data validation failed after processing"
            self.logger.error(msg)
            raise ValueError(msg)
        
        rows = len(next(iter(processed.values()))) if processed else 0
        total_elapsed = time.time() - start_time

        self.logger.info(
            f"Successfully loaded {rows} rows from "
            f"{len(filepaths) - len(failed_files)} file(s), "
            f"in {total_elapsed:.2f} seconds"
        )

        self.logger.debug(f"Data shape: {len(processed)} columns, {rows:,} rows")
        self.logger.debug(
            f"Memory usage per column: {self._estimate_memory_usage(processed)}"
        )

        return processed


    def _estimate_memory_usage(self, data: Dict[str, np.ndarray]) -> str:
        """Estimate memory usage of data dictionary"""
        try:
            total_bytes = sum(arr.nbytes for arr in data.values())
            if total_bytes < 1024**2:
                return f"{total_bytes / 1024:.1f} KB"

            elif total_bytes < 1024**3:
                return f"{total_bytes / 1024**2:.1f} MB"
            
            else:
                return f"{total_bytes / 1024**3:.2f} GB"
        
        except:
            return "unknown"


    def save(self,
        data: Dict[str, np.ndarray], 
        filepath: Union[str, Path],
        fmt: Optional[str]=None
    ):
        """
        Save processed data to a file.
        
        Args:
            data: Dictionary of numpy arrays to save
            filepath: Path where to save the file
            fmt: Optional format override (e.g., 'csv')
            
        Raises:
            ValueError: If format is unsupported or data is empty
        """
        if not data:
            error_msg = "Cannot save empty data"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        path = Path(filepath)
        rows = len(next(iter(data.values())))
        self.logger.info(f"💾 Saving {rows:,} rows to {path}")
        self.logger.debug(f"Data columns: {len(data)}")
        
        try:
            if fmt is None:
                handler = HandlerFactory.get_handler(path)
                self.logger.debug(
                    f"Using handler from file extension: ``{path.suffix}``"
                )

            else:
                if fmt not in HandlerFactory.HANDLERS:
                    error_msg = f"Unsupported format: {fmt}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                handler_class = HandlerFactory.HANDLERS[f".{fmt}"]
                handler = handler_class()
                self.logger.debug(f"Using explicit format handler: {fmt}")
            
            start_time = time.time()
            handler.save(data, path)
            elapsed = time.time() - start_time
            
            self.logger.info(
                f"✅ Successfully saved {rows:,} rows to {path} in {elapsed:.2f}s"
            )
            
        except Exception as e:
            error_msg = f"Failed to save data to {path}: {e}"
            self.logger.error(error_msg)
            raise


    def validate_data(self, data: Dict[str, np.ndarray]) -> bool:
        """
        Validate the structure and consistency of processed data.
        
        Args:
            data: Dictionary of numpy arrays to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        self.logger.debug("Starting data validation")
        
        if not data:
            self.logger.error("Data is empty")
            return False
        
        missing = [column for column in self.REQUIRED_COLUMNS if column not in data]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False
        
        lengths = {key: len(value) for key, value in data.items()}
        unique_lengths = set(lengths.values())
        
        if len(unique_lengths) > 1:
            self.logger.error(f"Inconsistent array lengths: {lengths}")
            return False
        
        zero_length = [key for key, value in data.items() if value.size == 0]
        if zero_length:
            self.logger.warning(f"Zero-length columns: {zero_length}")
        
        dtype_mismatch = []
        for key, value in data.items():
            if key in self.processor.SCHEMA:
                expected_dtype = self.processor.SCHEMA[key]["dtype"]
                if value.dtype != expected_dtype:
                    dtype_mismatch.append(f"{key}: {value.dtype} != {expected_dtype}")
        
        if dtype_mismatch:
            self.logger.warning(f"Dtype mismatches: {dtype_mismatch}")
        
        rows = len(next(iter(data.values())))
        self.logger.debug(
            f"✅ Data validation passed: {len(data)} columns, "
            f"{rows:,} rows"
        )
        return True



def validate_data(self, data: Dict[str, np.ndarray]) -> bool:
        """
        Validate the structure and consistency of processed data.
        
        Args:
            data: Dictionary of numpy arrays to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        self.logger.debug("Starting data validation")
        
        if not data:
            self.logger.error("Data is empty")
            return False
        
        missing = [column for column in self.REQUIRED_COLUMNS if column not in data]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False
        
        lengths = {key: len(value) for key, value in data.items()}
        unique_lengths = set(lengths.values())
        
        if len(unique_lengths) > 1:
            self.logger.error(f"Inconsistent array lengths: {lengths}")
            return False
        
        # Check for zero-length columns
        zero_length = [key for key, value in data.items() if value.size == 0]
        if zero_length:
            self.logger.warning(f"Zero-length columns: {zero_length}")
        
        # Check data types
        dtype_mismatch = []
        for key, value in data.items():
            if key in self.processor.SCHEMA:
                expected_dtype = self.processor.SCHEMA[key]["dtype"]
                if value.dtype != expected_dtype:
                    dtype_mismatch.append(f"{key}: {value.dtype} != {expected_dtype}")
        
        if dtype_mismatch:
            self.logger.warning(f"Dtype mismatches: {dtype_mismatch}")
        
        rows = len(next(iter(data.values())))
        self.logger.debug(
            f"✅ Data validation passed: {len(data)} columns, "
            f"{rows:,} rows"
        )
        return True



def load_dataset(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader()
    return loader.load(dataset)



def shuffle_and_split(
    data: Dict[str, np.ndarray], 
    val_ratio: float = 0.2, 
    test_ratio: float = 0.0,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], ...]:
    """
    Shuffle and split a dataset into train / validation / test subsets.
    Splits are all deterministic given the provided random ``seed`` into
    the ``NumPy`` and preserve the alignment across all the feature
    arrays.
    -----
    Args:
    data: Structured dataset with consistent array lengths.
    val_ratio: Fraction of data reserved for validation
    test_ratio: Fraction of data reserved for testing.
    seed: ``Numpy`` seed for reproducibility shuffling the data.
    --------
    Returns:
    Tuple of (``train``, ``validation``) if ``test_ratio == 0.0`` else \
        (``train``, ``validation``, ``test``).
    """
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"val_ratio must be between `0` and  `1`, got: {val_ratio}")
    
    if not 0.0 <= test_ratio <= 0.0:
        raise ValueError(f"test_ratio must lie in `0` and `1`, got: {test_ratio}")
    
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio = {val_ratio+test_ratio} > 1.0")

    lengths = {len(length) for length in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent array lengths: {lengths}")
    
    # Randomized sectoring of the dataset
    n_samples = next(iter(lengths))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    # Compute the splits
    val_split = int(n_samples * (1 - val_ratio - test_ratio))
    test_split = int(n_samples * (1 - test_ratio)) if test_ratio > 0 else n_samples

    ti = indices[:val_split]
    vi = indices[val_split:test_split]

    # Creates Splits
    train_data = {key: value[ti] for key, value in data.items()}
    val_data = {key: value[vi] for key, value in data.items()}

    if test_ratio > 0:
        test_idx = indices[test_split:]
        test_data = {key: value[test_idx] for key, value in data.items()}
        return train_data, val_data, test_dat
    
    return train_data, val_data
