import pytest
from env.grid_world import GridWorld, Position

def test_initialization():
    grid = GridWorld(10)
    assert grid.grid_size == 10

def test_invalid_initialization():
    with pytest.raises(ValueError):
        GridWorld(0)
    with pytest.raises(ValueError):
        GridWorld(-5)
    with pytest.raises(TypeError):
        GridWorld("10") # type: ignore

def test_valid_positions():
    grid = GridWorld(10)
    assert grid.is_valid_position(Position(0, 0)) is True
    assert grid.is_valid_position(Position(5, 5)) is True
    assert grid.is_valid_position(Position(9, 9)) is True

def test_invalid_positions():
    grid = GridWorld(10)
    assert grid.is_valid_position(Position(-1, 0)) is False
    assert grid.is_valid_position(Position(10, 10)) is False
    assert grid.is_valid_position(Position(0, -2)) is False
    
    with pytest.raises(TypeError):
        grid.is_valid_position((5, 5)) # type: ignore

def test_get_all_positions():
    grid = GridWorld(10)
    positions = grid.get_all_positions()
    
    assert len(positions) == 100
    assert Position(0, 0) in positions
    assert Position(9, 9) in positions
    
    # Ensure all returned positions are valid
    for pos in positions:
        assert grid.is_valid_position(pos)

def test_random_sampling():
    grid = GridWorld(10)
    
    # Verify validity of sampled positions
    for _ in range(10):
        pos = grid.sample_random_position()
        assert grid.is_valid_position(pos)
        
    # Verify deterministic behavior with seed
    p1 = grid.sample_random_position(seed=42)
    p2 = grid.sample_random_position(seed=42)
    assert p1 == p2
    
    assert p1 == Position(0, 7)
    assert grid.sample_random_position(seed=99) == Position(9, 5)
