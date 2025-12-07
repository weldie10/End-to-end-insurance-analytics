"""Tests for DataLoader class."""

import pytest
from pathlib import Path
from alphacare.data import DataLoader


def test_data_loader_initialization():
    """Test DataLoader initialization."""
    loader = DataLoader(data_path="Data")
    assert loader.data_path == Path("Data")
    assert loader.raw_data is None
    assert loader.processed_data is None


def test_data_loader_load_data():
    """Test loading data from file."""
    loader = DataLoader(data_path="Data")
    data = loader.load_data("MachineLearningRating_v3.txt")
    
    assert data is not None
    assert len(data) > 0
    assert "TotalPremium" in data.columns
    assert "TotalClaims" in data.columns


def test_data_loader_get_info():
    """Test getting data information."""
    loader = DataLoader(data_path="Data")
    loader.load_data("MachineLearningRating_v3.txt")
    info = loader.get_data_info()
    
    assert "shape" in info
    assert "columns" in info
    assert "missing_values" in info


def test_data_loader_preprocess():
    """Test data preprocessing."""
    loader = DataLoader(data_path="Data")
    loader.load_data("MachineLearningRating_v3.txt")
    processed = loader.preprocess_data()
    
    assert processed is not None
    assert len(processed) > 0

