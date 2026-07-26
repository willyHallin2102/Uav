"""
    database / handlers.py
    ----------------------
    Abstraction of handlers for the loader, this script contains all implementation
    of various handlers for potentially multiple data-types to abstract from the 
    loader which instead relies on this script to simply handling the file
    extensions and it handles the logic.
"""
from __future__ import annotations

import orjson
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv

from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Type, Union
from abc import ABC, abstractmethod

from logs.logger import Logger, Level



class FileHandler(ABC):
    """
    """

    def __init__(self, name: str="Handlers", level: Level = Level.INFO):
        """
            Initialize File Handler Instance
        """
        self.logger = Logger(name, level)
    

    def save(self, data: Dict[str, pd.DataFrame], filepath: Path) -> None:
        """
        """
        self.logger.debug(f"Starting save operation to {filepath}")

        if not data:
            self.logger.error("Attempt to save empty data directory")
            raise ValueError("Cannot save empty data directory")
        
        self.logger.debug("Attempt to save empty data directory")
        dataframe: pd.DataFrame = self._prepare_dataframe(data)

        if dataframe.empty:
            self.logger.error("Prepared dataframe is empty -- no data to save")
            raise ValueError("Prepared dataframe is empty")
        
        self.logger.debug(f"Writing dataframe with ``{len(dataframe)}`` rows to ``{filepath}``")
        self._write_dataframe(dataframe, filepath)
        self.logger.info(f"✅ Saved data to ``{filepath}``")
    

    @abstractmethod
    def load(self, filepath: Path, size: int) -> Iterator[pd.DataFrame]:
        """
        Yield ``pd.DataFrame`` chunks from loaded file.
        """
        raise NotImplementedError
    

    @abstractmethod
    def _write_dataframe(self, dataframe: pd.DataFrame, filepath: Path) -> pd.DataFrame:
        """
        """
        raise NotImplementedError
    

    def _prepare_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        """
        if not data: 
            self.logger.debug("Empty Data dictionary provided")
            return pd.DataFrame()
        
        self.logger.debug(f"Preparing dataframe with ``{len(data)}`` columns")
        df_dict: Dict[str, List[Any]] = {}

        for key, array in data.items():
            array = np.asarray(array)
            
            if array.size == 0:
                self.logger.debug(f"Column ``{key}`` is empty")
                df_dict[key] = []
                continue
            
            # Handle 1D numeric arrays efficiently
            if array.ndim == 1 and array.dtype != object:
                df_dict[key] = array.tolist()
                continue
            
            # Handle 2D+ arrays
            if array.ndim > 1:
                self.logger.debug(
                    f"Column ``{key}`` is {array.ndim}D,  "
                    "Converting to nested lists"
                )
                df_dict[key] = array.tolist()
                continue
            
            if array.dtype == object:

                self.logger.debug(
                    f"Column ``{key}`` is object dtype, processing individually"
                )
                column: List[Any] = [None] * len(array)
                for i, value in enumerate(array):

                    if isinstance(value, (list, np.ndarray, dict)):
                        column[i] = orjson.dumps(value).decode("utf-8")
                    
                    else:
                        column[i] = str(value)
                
                df_dict[key] = column
            
            else:
                df_dict[key] = array.tolist()
        
        result = pd.DataFrame(df_dict)
        self.logger.debug(
            f"Created dataframe with ``{len(result)}`` rows, "
            f"``{len(result.columns)}`` columns"
        )
        return result



class CsvHandler(FileHandler):
    """ Chunked CSV handler using PyArrow """

    ROW_SIZE        : Final[int] = 1024
    MIN_BLOCK_SIZE  : Final[int] = 1 << 20    # 1MB


    def __init__(self):
        super().__init__("CSV Handler")


    def load(self, filepath: Path, size: int) -> Iterator[pd.DataFrame]:
        """Implement the abstract load method - delegates to load_chunks"""
        return self.load_chunks(filepath, size)


    def load_chunks(self, filepath: Path, size: int) -> Iterator[pd.DataFrame]:
        """ Read CSV file in chunks using PyArrow """
        
        if not filepath.exists():
            message = f"CSV file ``{filepath}`` not found"
            self.logger.error(message)
            raise FileNotFoundError(message)
        
        self.logger.info(f"Loading CSV from ``{filepath}`` with chunk_size={size}")
        self.logger.debug(
            f"Filesize: ``{filepath.stat().st_size / (1024 ** 2):.2f}`` MB"
        )

        try:
            read_options = pcsv.ReadOptions(
                block_size=max(self.MIN_BLOCK_SIZE, size * self.ROW_SIZE), 
                use_threads=True
            )
            convert_options = pcsv.ConvertOptions(
                auto_dict_encode=False, 
                strings_can_be_null=True
            )

            self.logger.debug(
                f"Opening CSV with block_size={read_options.block_size}"
            )

            reader = pcsv.open_csv(
                filepath, 
                read_options=read_options, 
                convert_options=convert_options
            )

            batch_count = 0
            total_rows = 0

            for batch in reader:
                df = batch.to_pandas()

                if not df.empty:
                    batch_count += 1
                    total_rows += len(df)
                    self.logger.debug(
                        f"Yielded batch #{batch_count} with {len(df)} rows"
                    )
                    yield df

                else:
                    self.logger.warning(
                        f"Empty batch encountered at #({batch_count + 1})"
                    )

            self.logger.info(
                f"✅ Successfully loaded CSV from `{filepath}` "
                f"({batch_count} batches, {total_rows} rows)"
            )

        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            self.logger.error(f"Arrow error reading `{filepath}`: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error reading `{filepath}`: {e}")
            raise
    

    def _write_dataframe(self, df: pd.DataFrame, filepath: Path) -> None:
        """ Write `pd.DataFrame` to CSV """
        try:
            self.logger.debug(f"Writing {len(df)} rows to CSV at {filepath}")
            
            df.to_csv(filepath, index=False)

            self.logger.debug(f"✅ Successfully wrote {len(df)} rows to `{filepath}`")
            self.logger.info(f"Saved CSV with {len(df)} rows to `{filepath}`")

        except Exception as e:
            self.logger.error(f"Failed to write CSV to `{filepath}`: {e}")
            raise


class HandlerFactory:
    """ Factory for creating appropriate file-handler """
    HANDLERS: Dict[str, Type[FileHandler]] = {
        ".csv": CsvHandler, "csv": CsvHandler
    }
    _logger = Logger("Handler Factory", Level.INFO)


    @classmethod
    def get_handler(cls, path: Union[str, Path]) -> FileHandler:
        """ Get the appropriate file extension """
        path = Path(path)
        suffix = path.suffix.lower()

        if not suffix:
            raise ValueError(f"Cannot infer file type from file ``{path}``")

        handler_class = cls.HANDLERS.get(suffix)
        
        if not handler_class:    
            raise ValueError(
                f"Unsupported file type `{suffix}`, supported: "
                f"`{', '.join(cls.HANDLERS.keys())}`"
            )

        return handler_class()
