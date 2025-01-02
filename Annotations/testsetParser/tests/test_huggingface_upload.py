import pytest
import os
from datetime import datetime
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from testsetparser.state import State, BioAllowableActionTypes

def create_dummy_annotations():
    """Create dummy annotations for testing."""
    return [
        {
            "frame": 30,
            "time": 0.5,
            "action_type": BioAllowableActionTypes.LIQUID_HANDLING,
            "action_description": "Pipetting water into beaker",
            "detected_apparatus": "beaker, test tube",
            "detected_instruments": "pipette",
            "detected_materials": "water, salt solution",
            "spatial_information": '{"x": 100, "y": 200}'
        },
        {
            "frame": 60,
            "time": 1.0,
            "action_type": BioAllowableActionTypes.MIXING,
            "action_description": "Stirring the solution",
            "detected_apparatus": "beaker",
            "detected_instruments": "stirrer",
            "detected_materials": "mixed solution",
            "spatial_information": '{"x": 150, "y": 250}'
        }
    ]

@pytest.fixture
def state():
    """Create a State instance for testing."""
    s = State()
    s.video_url = "https://example.com/test_video.mp4"
    return s

def test_push_annotations_to_new_dataset():
    """Test pushing annotations to a new dataset."""
    # Setup
    state = State()
    state.video_url = "https://example.com/test_video.mp4"
    state.annotations = create_dummy_annotations()
    state.hf_dataset_repo = "LabARAgent/test_video_annotations"
    state.is_private_dataset = True
    
    # Execute
    state.push_to_huggingface()
    
    # Verify
    assert state.hf_error == "", f"Error occurred: {state.hf_error}"
    assert state.hf_success != ""
    assert "complete" in state.hf_progress.lower()

def test_push_additional_annotations():
    """Test pushing additional annotations to existing dataset."""
    # Setup
    state = State()
    state.video_url = "https://example.com/test_video2.mp4"
    state.annotations = create_dummy_annotations()
    state.hf_dataset_repo = "LabARAgent/test_video_annotations"
    state.is_private_dataset = True
    
    # Add different timestamps to make unique entries
    for ann in state.annotations:
        ann["time"] += 10.0
    
    # Execute
    state.push_to_huggingface()
    
    # Verify
    assert state.hf_error == "", f"Error occurred: {state.hf_error}"
    assert state.hf_success != ""
    assert "complete" in state.hf_progress.lower()

def test_empty_annotations():
    """Test error handling when no annotations exist."""
    # Setup
    state = State()
    state.annotations = []
    state.hf_dataset_repo = "LabARAgent/test_video_annotations"
    
    # Execute
    state.push_to_huggingface()
    
    # Verify
    assert "No annotations to push" in state.hf_error
    assert state.hf_success == ""

def test_invalid_repo_name():
    """Test error handling with invalid repository name."""
    # Setup
    state = State()
    state.annotations = create_dummy_annotations()
    state.hf_dataset_repo = "invalid/repo/name"  # Invalid format
    
    # Execute
    state.push_to_huggingface()
    
    # Verify
    assert state.hf_error != ""
    assert state.hf_success == ""

def test_progress_tracking():
    """Test progress message updates during upload."""
    # Setup
    state = State()
    state.video_url = "https://example.com/test_video3.mp4"
    state.annotations = create_dummy_annotations()
    state.hf_dataset_repo = os.getenv("HF_DATASET_REPO", "LabARAgent/test_video_annotations")
    state.is_private_dataset = True
    
    # Execute
    state.push_to_huggingface()
    
    # Verify progress messages
    assert state.hf_progress != ""
    assert any(msg in state.hf_progress.lower() for msg in ["preparing", "uploading", "complete"])
    assert state.hf_error == "", f"Error occurred: {state.hf_error}"
