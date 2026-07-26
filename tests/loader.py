"""
    tests / loader.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import io
import json
import numpy as np
import pandas as pd
import tempfile
import time
from typing import Any, Dict, List, Optional

from database.loader import DataLoader
from database.processors import DataProcessor
from logs.logger import Level
from tools.utilities import runner, builder, CommandSpec



# ======================================================================
#       Test Fixtures
# ======================================================================

def create_test_data(
    n_rows: int = 1000, include_all_columns: bool = True
) -> Dict[str, np.ndarray]:
    """
    Create synthetic test data matching the expected schema.
    
    Args:
        n_rows: Number of rows to generate
        include_all_columns: Whether to include all required columns
        
    Returns:
        Dictionary of numpy arrays
    """
    rng = np.random.default_rng(42)
    data = {
        "dvec": rng.random((n_rows, 3)).astype(np.float32),
        "rx_type": rng.integers(0, 3, size=n_rows).astype(np.uint8),
        "link_state": rng.integers(0, 2, size=n_rows).astype(np.uint8),
        "los_pl": rng.random(n_rows).astype(np.float32) * 100,
        "los_ang": rng.random((n_rows, 4)).astype(np.float32) * 360,
        "los_dly": rng.random(n_rows).astype(np.float32) * 50,
        "nlos_pl": rng.random((n_rows, 20)).astype(np.float32) * 100,
        "nlos_ang": rng.random((n_rows, 20, 4)).astype(np.float32) * 360,
        "nlos_dly": rng.random((n_rows, 20)).astype(np.float32) * 50,
    }
    
    if not include_all_columns:
        cols_to_remove = ["nlos_pl", "nlos_ang"]
        for column in cols_to_remove:

            if column in data:
                del data[column]
    
    return data


def create_test_csv(
    n_rows: int = 1000, include_all_columns: bool = True,
    filepath: Optional[Path] = None
) -> Path:
    """
    Create a test CSV file with synthetic data.
    
    Args:
        n_rows: Number of rows to generate
        include_all_columns: Whether to include all required columns
        filepath: Optional path to save the CSV
        
    Returns:
        Path to the created CSV file
    """
    if filepath is None:
        filepath = Path(tempfile.mkdtemp()) / "test_data.csv"
    
    data = create_test_data(n_rows, include_all_columns)
    
    df_dict = {}
    for key, value in data.items():
    
        if value.ndim == 1:
            df_dict[key] = value.tolist()
    
        elif value.ndim == 2:
            # Convert 2D arrays to JSON strings for CSV storage
            df_dict[key] = [json.dumps(row.tolist()) for row in value]
    
        elif value.ndim == 3:
            # Convert 3D arrays to JSON strings for CSV storage
            df_dict[key] = [json.dumps(row.tolist()) for row in value]
    
    df = pd.DataFrame(df_dict)
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    
    return filepath


def create_test_directory(n_files: int = 3, n_rows_per_file: int = 500) -> Path:
    """
    Create a directory with multiple test CSV files.
    
    Args:
        n_files: Number of CSV files to create
        n_rows_per_file: Number of rows per file
        
    Returns:
        Path to the test directory
    """
    tmp = Path(tempfile.mkdtemp()) / "test_datasets"
    tmp.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_files):
        filepath = tmp / f"test_data_{i}.csv"
        create_test_csv(n_rows_per_file, True, filepath)
    
    return tmp


# ======================================================================
#       Test Helpers
# ======================================================================

def get_dataloader(
    n_workers: Optional[int] = None, chunk_size: int = 100,
    level: Level = Level.DEBUG
) -> DataLoader:
    """
    Create a fresh DataLoader instance for testing.
    """
    return DataLoader(
        n_workers=n_workers, chunk_size=chunk_size,
        prefer_processes=False, level=level
    )


# ======================================================================
#       Test Methods
# ======================================================================

def test_initialization(args: argparse.Namespace):
    """Test DataLoader initialization and configuration."""
    print("1. ---------- Testing DataLoader Initialization ----------")
    
    loader = get_dataloader()
    print(f"DataLoader instance: {loader}")
    print(f"Data directory: {loader.directory}")
    print(f"Number of workers: {loader.n_workers}")
    print(f"Chunk size: {loader.chunk_size}")
    print(f"Required columns: {loader.REQUIRED_COLUMNS}")
    
    print("\n2. ---------- Testing custom configuration ----------")
    loader_custom = DataLoader(
        n_workers=4, chunk_size=5000, prefer_processes=True, level=Level.DEBUG
    )
    print(f"Custom workers: {loader_custom.n_workers}")
    print(f"Custom chunk size: {loader_custom.chunk_size}")
    print(f"Prefer processes: {loader_custom.prefer_processes}")
    
    print("\n3. ---------- Testing directory creation ----------")
    d = Path(tempfile.mkdtemp()) / "custom_datasets"
    loader_custom.directory = d
    loader_custom.directory.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {d.exists()}")
    
    print("\n✅ Initialization test passed")


def test_data_generation(args: argparse.Namespace):
    """Test synthetic data generation for testing."""
    print("1. ---------- Testing Data Generation ----------")
    
    n_rows = 100
    data = create_test_data(n_rows)
    
    print(f"Generated {len(data)} columns with {n_rows} rows")
    for key, value in data.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    print("\n2. ---------- Testing CSV creation ----------")
    with tempfile.TemporaryDirectory() as tmp:
        path = create_test_csv(n_rows, True, Path(tmp) / "test.csv")
        print(f"Created CSV at: {path}")
        print(f"File size: {path.stat().st_size / 1024:.2f} KB")
        
        df = pd.read_csv(path)
        print(f"CSV has {len(df)} rows and {len(df.columns)} columns")
        assert len(df) == n_rows, f"Expected {n_rows} rows, got {len(df)}"
    
    print("\n3. ---------- Testing multi-file directory creation ----------")
    with tempfile.TemporaryDirectory():
        path = create_test_directory(3, 100)
        files = list(path.glob("*.csv"))
        print(f"Created {len(files)} CSV files in {path}")
        
        for f in files:
            print(f"  {f.name}: {f.stat().st_size / 1024:.2f} KB")
    
    print("\n✅ Data generation test passed")


def test_load_single_file(args: argparse.Namespace):
    """Test loading a single CSV file."""
    print("1. ---------- Testing Single File Loading ----------")
    
    n_rows = 200
    with tempfile.TemporaryDirectory() as tmp:
        path = create_test_csv(n_rows, True, Path(tmp) / "test.csv")
        loader = get_dataloader(chunk_size=50)
        
        print(f"Loading file: {path}")
        start_time = time.time()
        data = loader.load(str(path))
        elapsed = time.time() - start_time
        
        print(f"Loaded {len(data)} columns")
        rows = len(next(iter(data.values())))
        print(f"Total rows: {rows}")
        print(f"Loading time: {elapsed:.3f}s")
        
        for key, value in data.items():
            print(f"\t{key}: shape={value.shape}, dtype={value.dtype}")
        
        assert rows == n_rows, f"Expected {n_rows} rows, got {rows}"
        assert all(col in data for col in loader.REQUIRED_COLUMNS), \
            "Missing required columns"
    
    print("\n2. ---------- Testing with absolute path ----------")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "absolute.csv"
        
        create_test_csv(50, True, path)
        
        loader = get_dataloader()
        data = loader.load(path)
        rows = len(next(iter(data.values())))
        print(f"Loaded {rows} rows from absolute path")
        assert rows == 50, f"Expected 50 rows, got {rows}"
    
    print("\n✅ Single file load test passed")



def test_load_single_dataset(args: argparse.Namespace):

    loader = get_dataloader()
    dataset_path = "uav_london/train.csv"
    
    print(f"Loading dataset: {dataset_path}")

    try:
        start_time = time.time()
        data = loader.load(dataset_path)
        elapsed = time.time() - start_time

        assert data is not None, "Data returned is None"
        assert isinstance(data, dict), "Data is not a dictionary"
        
        rows = len(next(iter(data.values())))
        print(f"\n✅ Loaded {rows:,} rows in {elapsed:.2f}s")

    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        print("  Skipping test - dataset may not be available")


def test_load_multiple_files(args: argparse.Namespace):
    """Test loading multiple CSV files."""
    print("1. ---------- Testing Multiple File Loading ----------")
    
    with tempfile.TemporaryDirectory():

        n_files = 3
        n_rows_per_file = 100
        path = create_test_directory(n_files, n_rows_per_file)
        paths = list(path.glob("*.csv"))
        
        loader = get_dataloader(chunk_size=50)
        print(f"Loading {len(paths)} files...")
        
        start_time = time.time()
        data = loader.load([str(p) for p in paths])
        elapsed = time.time() - start_time
        
        total_rows = len(next(iter(data.values())))
        expected_rows = n_files * n_rows_per_file
        print(f"Loaded {total_rows} rows from {len(paths)} files")
        print(f"Expected: {expected_rows} rows")
        print(f"Loading time: {elapsed:.3f}s")
        
        assert total_rows == expected_rows, \
            f"Expected {expected_rows}, got {total_rows}"
        
        missing = [c for c in loader.REQUIRED_COLUMNS if c not in data]
        assert not missing, f"Missing columns: {missing}"
    
    print("\n2. ---------- Testing with mixed valid/invalid files ----------")
    with tempfile.TemporaryDirectory() as tmp:
        valid_path = create_test_csv(50, True, Path(tmp) / "valid.csv")
        invalid_path = Path(tmp) / "nonexistent.csv"
        
        loader = get_dataloader()
        
        try:
            data = loader.load([str(valid_path), str(invalid_path)])
            rows = len(next(iter(data.values())))
            print(f"Loaded {rows} rows from valid file, skipped invalid")
            assert rows == 50, f"Expected 50 rows, got {rows}"
        
        except RuntimeError as e:
            print(f"Expected error: {e}")
    
    print("\n✅ Multiple file load test passed")


def test_load_multiple_dataset(args: argparse.Namespace):
    loader = get_dataloader()
    datasets = [
        "uav_beijing/train.csv",
        "uav_boston/train.csv", 
        "uav_london/train.csv",
        "uav_moscow/train.csv",
        "uav_tokyo/train.csv"
    ]

    print(f"Loading {len(datasets)} datasets...")
    start_time = time.time()
    try:
        data = loader.load(datasets)
        elapsed = time.time() - start_time

        assert data is not None, "Data returned is None"
        assert isinstance(data, dict), "Data is not a dictionary"

        missing = [c for c in loader.REQUIRED_COLUMNS if c not in data]
        assert not missing, f"Missing required columns: {missing}"
        
        rows = len(next(iter(data.values())))
        print(
            f"\n✅ Loaded {rows:,} total rows from {len(datasets)} "
            f"files in {elapsed:.2f}s"
        )

    except FileNotFoundError as e:
        print(f"❌ Dataset(s) not found: {e}")
        print("  Skipping test - datasets may not be available")


def test_missing_columns(args: argparse.Namespace):
    """Test handling of missing columns in data."""
    print("1. ---------- Testing Missing Column Handling ----------")
    
    with tempfile.TemporaryDirectory() as tmp:

        path = create_test_csv(100, False, Path(tmp) / "missing.csv")
        loader = get_dataloader(chunk_size=50)
        
        print(f"Loading file with missing columns: {path}")
        try:
            data = loader.load(str(path))
            rows = len(next(iter(data.values())))
            print(f"Loaded {rows} rows")
            
            missing = [c for c in loader.REQUIRED_COLUMNS if c not in data]
            print(f"Missing columns in loaded data: {missing}")
        
        except Exception as e:
            print(f"Error (expected): {e}")
    
    print("\n2. ---------- Testing empty data handling ----------")
    with tempfile.TemporaryDirectory() as tmp:
        
        empty_path = Path(tmp) / "empty.csv"
        pd.DataFrame(columns=loader.REQUIRED_COLUMNS).to_csv(empty_path, index=False)
        
        loader = get_dataloader()
        try:
            data = loader.load(str(empty_path))
            rows = len(next(iter(data.values())))
            print(f"Loaded {rows} rows from empty file")
            assert rows == 0, f"Expected 0 rows, got {rows}"

        except Exception as e:
            print(f"Error (expected): {e}")
    
    print("\n✅ Missing columns test passed")


def test_data_processing(args: argparse.Namespace):
    """Test data processing and transformation."""
    print("1. ---------- Testing Data Processing ----------")
    
    n_rows = 100
    with tempfile.TemporaryDirectory() as tmp:
        loader = get_dataloader(chunk_size=50)
        data = loader.load(str(create_test_csv(n_rows, True, Path(tmp) / "test.csv")))
        
        print("Verifying data types:")
        for key, value in data.items():
            expected_dtype = loader.processor.SCHEMA[key]["dtype"]
            actual_dtype = value.dtype
            print(f"  {key}: expected={expected_dtype}, actual={actual_dtype}")
    
            if not key.endswith("ang") and not key.endswith("pl"):
                assert value.dtype == expected_dtype, f"Wrong dtype for {key}"
        
        print("\nVerifying shapes:")
        for key, value in data.items():
            shape = value.shape
            print(f"  {key}: shape={shape}")
        
        print("\nVerifying stacked columns:")
        columns = ["dvec", "los_ang", "nlos_pl", "nlos_ang", "nlos_dly"]
        for column in columns:
            if column in data:
                expected_dim = loader.processor.SCHEMA[column].get("dim")
                actual_dim = data[column].shape[1:] \
                    if len(data[column].shape) > 1 else None
                print(
                    f"{column}: expected_dim={expected_dim}, actual_dim={actual_dim}"
                )
    
    print("\n✅ Data processing test passed")


def test_data_validation(args: argparse.Namespace):
    """Test data validation functionality."""
    print("1. ---------- Testing Data Validation ----------")
    
    loader = get_dataloader()
    
    data = create_test_data(100, True)
    is_valid = loader.validate_data(data)

    print(f"Valid data validation: {is_valid}")
    assert is_valid, "Valid data should pass validation"
    
    # Test with empty data
    empty = {}
    is_valid = loader.validate_data(empty)
    print(f"Empty data validation: {is_valid}")
    assert not is_valid, "Empty data should fail validation"
    
    # Test with missing columns
    missing = create_test_data(100, False)
    is_valid = loader.validate_data(missing)
    print(f"Missing columns validation: {is_valid}")
    assert not is_valid, "Missing columns should fail validation"
    
    inconsistent_data = create_test_data(100, True)
    
    # Make one column shorter
    inconsistent_data["rx_type"] = inconsistent_data["rx_type"][:50]
    is_valid = loader.validate_data(inconsistent_data)
    print(f"Inconsistent lengths validation: {is_valid}")
    assert not is_valid, "Inconsistent lengths should fail validation"
    
    zero_data = create_test_data(0, True)
    is_valid = loader.validate_data(zero_data)
    print(f"Zero-length data validation: {is_valid}")
    
    print("\n✅ Data validation test passed")


def test_save(args: argparse.Namespace):
    """Test saving data to file."""
    print("1. ---------- Testing Save Functionality ----------")
    
    with tempfile.TemporaryDirectory() as tmp:
        loader = get_dataloader()
        data = create_test_data(100, True)
        path = Path(tmp) / "saved_data.csv"
        
        print(f"Saving data to: {path}")
        loader.save(data, path)
        
        assert path.exists(), "Save file was not created"
        filesize = path.stat().st_size
        print(f"File size: {filesize / 1024:.2f} KB")
        
        print("Loading saved file...")
        loaded_data = loader.load(str(path))
        rows = len(next(iter(loaded_data.values())))
        print(f"Loaded {rows} rows from saved file")
        assert rows == 100, f"Expected 100 rows, got {rows}"
    
    print("\n2. ---------- Testing save with explicit format ----------")
    with tempfile.TemporaryDirectory() as tmp:
        loader = get_dataloader()
        data = create_test_data(50, True)
        path = Path(tmp) / "explicit.csv"
        
        loader.save(data, path, fmt="csv")
        assert path.exists(), "File not created with explicit format"
        print(f"Explicit format save successful: {path}")
    
    print("\n3. ---------- Testing save with empty data ----------")
    with tempfile.TemporaryDirectory() as tmp:
        loader = get_dataloader()
        empty = {}
        path = Path(tmp) / "empty.csv"
        
        try:
            loader.save(empty, path)
            print("Error: Should not save empty data")
        
        except ValueError as e:
            print(f"Error (expected): {e}")
    
    print("\n✅ Save functionality test passed")


def test_chunk_processing(args: argparse.Namespace):
    """Test chunked processing of large files."""
    print("1. ---------- Testing Chunk Processing ----------")
    
    n_rows = 1000
    size = 100
    
    with tempfile.TemporaryDirectory() as tmp:
        path = create_test_csv(n_rows, True, Path(tmp) / "large.csv")
        loader = get_dataloader(chunk_size=size)
        
        print(f"Loading {n_rows} rows with chunk size {size}")
        start_time = time.time()
        data = loader.load(str(path))
        elapsed = time.time() - start_time
        
        rows = len(next(iter(data.values())))
        print(f"Loaded {rows} rows in {elapsed:.3f}s")
        print(f"Expected {n_rows} rows")
        assert rows == n_rows, f"Expected {n_rows}, got {rows}"
    
    print("\n2. ---------- Testing various chunk sizes ----------")
    chunk_sizes = [50, 200, 500]
    
    with tempfile.TemporaryDirectory() as tmp:
        path = create_test_csv(500, True, Path(tmp) / "test.csv")
        
        for size in chunk_sizes:
            loader = get_dataloader(chunk_size=size)
            start_time = time.time()
            data = loader.load(str(path))
            elapsed = time.time() - start_time
            
            rows = len(next(iter(data.values())))
            print(f"Chunk size {size}: {rows} rows in {elapsed:.3f}s")
            assert rows == 500, f"Expected 500 rows, got {rows}"
     
    print("\n✅ Chunk processing test passed")


def test_parallel_processing(args: argparse.Namespace):
    """Test parallel processing with multiple workers."""
    print("1. ---------- Testing Parallel Processing ----------")
    
    n_rows = 500
    chunk_size = 50
    
    with tempfile.TemporaryDirectory():
        
        path = create_test_directory(3, n_rows)
        paths = list(path.glob("*.csv"))
        
        print(f"Processing {len(paths)} files with {n_rows} rows each")
        
        loader_single = get_dataloader(n_workers=1, chunk_size=chunk_size)
        
        start_time = time.time()
        data_single = loader_single.load([str(p) for p in paths])
        single_time = time.time() - start_time
        single_rows = len(next(iter(data_single.values())))
        print(f"Single worker: {single_rows} rows in {single_time:.3f}s")
        
        loader_multi = get_dataloader(n_workers=3, chunk_size=chunk_size)
        start_time = time.time()
        data_multi = loader_multi.load([str(p) for p in paths])
        multi_time = time.time() - start_time
        multi_rows = len(next(iter(data_multi.values())))
        print(f"Multi workers (3): {multi_rows} rows in {multi_time:.3f}s")
        
        print(f"Speedup: {single_time / multi_time:.2f}x")
        
        for key in loader_single.REQUIRED_COLUMNS:
            if key in data_single and key in data_multi:
                assert len(data_single[key]) == len(data_multi[key]), \
                    f"Length mismatch for {key}"
                print(f"  {key}: {len(data_single[key])} rows (consistent)")
    
    print("\n2. ---------- Testing thread vs process ----------")
    with tempfile.TemporaryDirectory() as tmp:
        path = create_test_directory(2, 200)
        paths = list(path.glob("*.csv"))
        
        # Thread-based
        thread = DataLoader(n_workers=2, chunk_size=50, prefer_processes=False)

        start_time = time.time()
        
        data_thread = thread.load([str(p) for p in paths])
        thread_time = time.time() - start_time
        thread_rows = len(next(iter(data_thread.values())))
        
        print(f"Thread-based: {thread_rows} rows in {thread_time:.3f}s")
        
        proc = DataLoader(n_workers=2, chunk_size=50, prefer_processes=True)

        start_time = time.time()
        data_process = proc.load([str(p) for p in paths])
        process_time = time.time() - start_time
        process_rows = len(next(iter(data_process.values())))
        print(f"Process-based: {process_rows} rows in {process_time:.3f}s")
        
        print(f"Thread vs Process: {thread_time / process_time:.2f}x")
    
    print("\n✅ Parallel processing test passed")


def test_processor_integration(args: argparse.Namespace):
    """Test DataProcessor integration and schema handling."""
    print("1. ---------- Testing Processor Integration ----------")
    
    processor = DataProcessor()
    print(f"Processor schema has {len(processor.SCHEMA)} columns")
    
    n_rows = 50
    data = create_test_data(n_rows, True)
    df_dict = {}
    for key, value in data.items():
        if value.ndim == 1:
            df_dict[key] = value.tolist()
        else:
            # For 2D+ arrays, convert to strings for DataFrame
            df_dict[key] = [json.dumps(row.tolist()) for row in value]
    
    df = pd.DataFrame(df_dict)
    
    print(f"Processing DataFrame with {len(df)} rows and {len(df.columns)} columns")
    processed = processor.process(df)
    print(f"Processed {len(processed)} columns")
    
    for key, value in processed.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    print("\n2. ---------- Testing concatenation ----------")
    chunk1 = processor.process(df)
    chunk2 = processor.process(df)
    
    concatenated = processor.concatenate([chunk1, chunk2])
    rows = len(next(iter(concatenated.values())))
    print(f"Concatenated {len(concatenated)} columns with {rows} rows")
    assert rows == n_rows * 2, f"Expected {n_rows * 2} rows, got {rows}"
    
    print("\n✅ Processor integration test passed")


# ======================================================================
#       Main Runner
# ======================================================================

COMMON = [
    {"flags": ["--n-samples", "-n"], "kwargs": {"type": int, "default": 100}},
    {"flags": ["--verbose", "-v"], "kwargs": {"action": "store_true"}},
]
TEST_ARGS = [*COMMON]


@runner
def main():
    p = builder([
        CommandSpec(
            "init",
            "Test DataLoader initialization and configuration",
            test_initialization,
            TEST_ARGS
        ),
        CommandSpec(
            "data_gen",
            "Test synthetic data generation for testing",
            test_data_generation,
            TEST_ARGS
        ),
        CommandSpec(
            "load_single",
            "Test loading a single CSV file",
            test_load_single_file,
            TEST_ARGS
        ),
        CommandSpec(
            "load_single_real",
            "Test loading a single CSV file",
            test_load_single_dataset,
            TEST_ARGS
        ),
        CommandSpec(
            "load_multiple",
            "Test loading multiple CSV files",
            test_load_multiple_files,
            TEST_ARGS
        ),
        CommandSpec(
            "load_multiple_real",
            "Test loading multiple CSV files",
            test_load_multiple_dataset,
            TEST_ARGS
        ),
        CommandSpec(
            "missing_cols",
            "Test handling of missing columns in data",
            test_missing_columns,
            TEST_ARGS
        ),
        CommandSpec(
            "processing",
            "Test data processing and transformation",
            test_data_processing,
            TEST_ARGS
        ),
        CommandSpec(
            "validation",
            "Test data validation functionality",
            test_data_validation,
            TEST_ARGS
        ),
        CommandSpec(
            "save",
            "Test saving data to file",
            test_save,
            TEST_ARGS
        ),
        CommandSpec(
            "chunks",
            "Test chunked processing of large files",
            test_chunk_processing,
            TEST_ARGS
        ),
        CommandSpec(
            "parallel",
            "Test parallel processing with multiple workers",
            test_parallel_processing,
            TEST_ARGS
        ),
        CommandSpec(
            "processor",
            "Test DataProcessor integration and schema handling",
            test_processor_integration,
            TEST_ARGS
        ),
    ])

    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
