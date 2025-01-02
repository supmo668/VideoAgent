from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime

import reflex as rx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from datasets import Dataset, load_dataset
import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.utils import RepositoryNotFoundError
import tempfile

# Import from app/models
import sys
from _PATH import EXTRA_PATHS
sys.path.extend(EXTRA_PATHS)
from models.llm.action_steps import ActionProtocol, BioAllowableActionTypes

class State(rx.State):
    """The app state."""
    # Video state
    video_url: str = ""
    video_error: str = ""
    current_time: float = 0
    fps: int = 60  # default fps
    
    # Form state
    is_natural_language_mode: bool = False
    natural_language_description: str = ""
    action_type: str = "LIQUID_HANDLING"
    action_description: str = ""
    detected_apparatus: str = ""  # Comma-separated string
    detected_instruments: str = ""  # Comma-separated string
    detected_materials: str = ""  # Comma-separated string
    spatial_information: str = ""  # JSON string
    
    # Annotations
    annotations: List[Dict[str, Any]] = []
    table_columns: List[str] = [
        "Frame",
        "Time (s)",
        "Action Type",
        "Description",
        "Apparatus",
        "Instruments",
        "Materials",
    ]

    # Hugging Face integration
    hf_dataset_repo: str = "LabARAgent/labAR_video_reporting"
    is_private_dataset: bool = True
    hf_error: str = ""
    hf_success: str = ""
    hf_progress: str = ""  # New field for progress updates

    def validate_video_url(self, url: str) -> bool:
        """Validate video URL format and accessibility."""
        try:
            if not url:
                self.video_error = "Please enter a video URL"
                return False
                
            # Check URL format
            from urllib.parse import urlparse
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                self.video_error = "Invalid URL format"
                return False
            
            # Check if URL ends with common video extensions
            video_extensions = ['.mp4', '.webm', '.ogg', '.mov']
            if not any(url.lower().endswith(ext) for ext in video_extensions):
                self.video_error = "URL must point to a video file (mp4, webm, ogg, mov)"
                return False
            
            # TODO: Add actual video file accessibility check if needed
            return True
            
        except Exception as e:
            self.video_error = f"Error validating URL: {str(e)}"
            return False
    
    def set_video_url(self, url: str):
        """Set the video URL with validation."""
        self.video_error = ""  # Clear previous errors
        if self.validate_video_url(url):
            self.video_url = url
        else:
            self.video_url = ""  # Clear invalid URL
    
    def update_progress(self, progress: dict):
        """Update the current time from progress data."""
        try:
            self.current_time = progress["playedSeconds"]
        except (KeyError, TypeError) as e:
            self.video_error = f"Error updating video progress: {str(e)}"
    
    def set_fps(self, value: str):
        """Set the FPS value with validation."""
        try:
            fps = int(value)
            if fps <= 0:
                self.video_error = "FPS must be greater than 0"
            else:
                self.fps = fps
                self.video_error = ""
        except ValueError:
            self.video_error = "FPS must be a valid number"
    
    def validate_hf_repo(self) -> bool:
        """Validate Hugging Face repository name."""
        if not self.hf_dataset_repo:
            self.hf_error = "Repository name is required"
            return False
            
        # Check format (username/dataset-name)
        parts = self.hf_dataset_repo.split('/')
        if len(parts) != 2 or not all(parts):
            self.hf_error = "Repository must be in format 'username/dataset-name'"
            return False
            
        return True
    
    def _process_string_list(self, value: str | list, field_name: str) -> list:
        """Process a value that should be a list of strings.
        
        Args:
            value: String with comma-separated values or list
            field_name: Name of the field for error messages
            
        Returns:
            List of strings
        """
        try:
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
            if not value:  # Handle empty string
                return []
            return [str(x.strip()) for x in value.split(",") if x.strip()]
        except Exception as e:
            print(f"Error processing {field_name}: {value} of type {type(value)}")
            print(f"Error details: {str(e)}")
            return []

    def _process_annotation(self, ann: dict) -> dict:
        """Process a single annotation into the correct format.
        
        Args:
            ann: Raw annotation dictionary
            
        Returns:
            Processed annotation dictionary
        """
        try:
            # Process lists
            apparatus = self._process_string_list(ann["detected_apparatus"], "detected_apparatus")
            instruments = self._process_string_list(ann["detected_instruments"], "detected_instruments")
            materials = self._process_string_list(ann["detected_materials"], "detected_materials")
            
            # Process spatial information
            spatial_info = ann["spatial_information"]
            if isinstance(spatial_info, str):
                try:
                    spatial_info = json.loads(spatial_info)
                except json.JSONDecodeError:
                    spatial_info = {}
            
            return {
                "frame": int(ann["frame"]),
                "timestamp": float(ann["time"]),
                "action_type": str(ann["action_type"]),
                "action_description": str(ann["action_description"]),
                "detected_apparatus": apparatus,
                "detected_instruments": instruments,
                "detected_materials": materials,
                "spatial_information": str(spatial_info),
                "video_url": str(self.video_url),
                "upload_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error processing annotation: {ann}")
            print(f"Error details: {str(e)}")
            raise

    def _create_or_get_dataset(self, api: HfApi) -> bool:
        """Create dataset if it doesn't exist.
        
        Args:
            api: Hugging Face API instance
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.set_hf_progress("Checking dataset existence...")
            try:
                api.dataset_info(self.hf_dataset_repo)
                self.set_hf_progress("Dataset found. Preparing to add new data...")
            except RepositoryNotFoundError:
                self.set_hf_progress("Dataset not found. Creating new dataset...")
                api.create_repo(
                    repo_id=self.hf_dataset_repo,
                    repo_type="dataset",
                    private=self.is_private_dataset
                )
            return True
        except Exception as e:
            self.hf_error = f"Error creating/accessing dataset: {str(e)}"
            return False

    def _create_data_file(self, processed_annotations: list) -> tuple[str, bool]:
        """Create temporary JSON file with annotations.
        
        Args:
            processed_annotations: List of processed annotations
            
        Returns:
            tuple: (temp_file_path, success)
        """
        try:
            self.set_hf_progress("Preparing data for upload...")
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump({"annotations": processed_annotations}, f, indent=2)
                return f.name, True
        except Exception as e:
            self.hf_error = f"Error creating data file: {str(e)}"
            return "", False

    def push_to_huggingface(self) -> None:
        """Push annotations to Hugging Face dataset with enhanced error handling."""
        self.hf_error = ""
        self.hf_success = ""
        self.hf_progress = ""
        temp_file = None
        
        try:
            # Validate inputs
            if not self.validate_hf_repo():
                return
            
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                self.hf_error = "HUGGINGFACE_TOKEN not found in environment variables"
                return
            
            if not self.annotations:
                self.hf_error = "No annotations to push"
                return
            
            # Initialize HF API
            api = HfApi(token=token)
            
            # Create or get dataset
            if not self._create_or_get_dataset(api):
                return
            
            # Process annotations
            self.set_hf_progress("Processing annotations...")
            try:
                processed_annotations = [
                    self._process_annotation(ann) for ann in self.annotations
                ]
            except Exception as e:
                self.hf_error = f"Error processing annotations: {str(e)}"
                return
            
            # Create temporary file
            temp_file, success = self._create_data_file(processed_annotations)
            if not success:
                return
            
            # Create and execute commit
            try:
                self.set_hf_progress("Creating commit operation...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"annotations_{timestamp}.json"
                
                operations = [
                    CommitOperationAdd(
                        path_in_repo=f"data/{filename}",
                        path_or_fileobj=temp_file
                    )
                ]
                
                self.set_hf_progress("Uploading new annotations...")
                api.create_commit(
                    repo_id=self.hf_dataset_repo,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Add new annotations batch - {len(processed_annotations)} items"
                )
                
                self.hf_success = f"Successfully pushed {len(processed_annotations)} annotations to {self.hf_dataset_repo}"
                self.set_hf_progress("Upload complete!")
                
            except Exception as e:
                self.hf_error = f"Error creating commit: {str(e)}"
                
        except Exception as e:
            self.hf_error = f"Error pushing to Hugging Face: {str(e)}"
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            if self.hf_error:
                self.set_hf_progress("Upload failed!")
    
    def set_hf_progress(self, message: str):
        """Update progress message."""
        self.hf_progress = message
    
    def toggle_form_mode(self):
        """Toggle between structured and natural language form modes."""
        self.is_natural_language_mode = not self.is_natural_language_mode
    
    def set_natural_language_description(self, value: str):
        """Set the natural language description."""
        self.natural_language_description = value
    
    def set_action_type(self, value: str):
        self.action_type = value
            
    def set_action_description(self, value: str):
        self.action_description = value
        
    def set_detected_apparatus(self, value: str):
        self.detected_apparatus = value
        
    def set_detected_instruments(self, value: str):
        self.detected_instruments = value
        
    def set_detected_materials(self, value: str):
        self.detected_materials = value
            
    def set_spatial_information(self, value: str):
        self.spatial_information = value
    
    def add_annotations(self):
        """Add current form data as an annotation."""
        annotation = {
            "frame": self.current_frame,
            "time": self.current_time,
            "action_type": self.action_type,
            "action_description": self.action_description,
            "detected_apparatus": [x.strip() for x in self.detected_apparatus.split(",") if x.strip()] if self.detected_apparatus else [],
            "detected_instruments": [x.strip() for x in self.detected_instruments.split(",") if x.strip()] if self.detected_instruments else [],
            "detected_materials": [x.strip() for x in self.detected_materials.split(",") if x.strip()] if self.detected_materials else [],
            "spatial_information": self.spatial_information
        }
        self.annotations.append(annotation)
        # Clear form fields after adding annotation
        # self.clear_form()
    
    def clear_form(self):
        """Clear form fields."""
        self.action_description = ""
        self.detected_apparatus = ""
        self.detected_instruments = ""
        self.detected_materials = ""
        self.spatial_information = ""

    def submit_form_fields(self):
        """Process the form fields based on the current mode."""
        if self.is_natural_language_mode:
            print(f"Processing natural language description: {self.natural_language_description}")
        else:
            print(f"Processing structured form fields:")
            print(f"Action type: {self.action_type}")
            print(f"Action description: {self.action_description}")
            print(f"Detected apparatus: {self.detected_apparatus}")
            print(f"Detected instruments: {self.detected_instruments}")
            print(f"Detected materials: {self.detected_materials}")
            print(f"Spatial information: {self.spatial_information}")

    @rx.var
    def current_frame(self) -> int:
        """Calculate current frame from time."""
        return int(self.current_time * self.fps)
    
    @rx.var
    def table_data(self) -> List[List[Any]]:
        """Convert annotations to table data format."""
        return [
            [
                ann["frame"],
                f"{ann['time']:.2f}",
                ann["action_type"],
                ann["action_description"],
                ", ".join(ann["detected_apparatus"]),
                ", ".join(ann["detected_instruments"]),
                ", ".join(ann["detected_materials"]),
            ]
            for ann in self.annotations
        ]

    def set_hf_dataset_repo(self, repo: str):
        """Set the Hugging Face dataset repository."""
        self.hf_dataset_repo = repo
    
    def set_dataset_privacy(self, is_private: bool):
        """Set whether the dataset should be private."""
        self.is_private_dataset = is_private
