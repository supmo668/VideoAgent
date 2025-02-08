from pydantic import BaseModel, Field, field_validator
from typing import List, Union, Optional, Dict, Any

from enum import Enum

class BioAllowableActionTypes(str, Enum):
    """
    Enum representing action types specific to biological science laboratories.
    """
    LIQUID_HANDLING = "LIQUID_HANDLING"
    INCUBATION = "INCUBATION"
    MIXING = "MIXING"
    SEPARATION = "SEPARATION"
    OBSERVATION = "OBSERVATION"
    PLATE_HANDLING = "PLATE_HANDLING"
    STORAGE = "STORAGE"
    LABELLING = "LABELLING"
    INSTRUMENTATION = "INSTRUMENTATION"
    EXTRACTION = "EXTRACTION"
    PIPETTING = "PIPETTING"

    def __str__(self):
        return self.value

    def __int__(self):
        return list(self.__class__).index(self) + 1

class StepParameters(BaseModel):
    """
    Represents the parameters for a single step in the customized protocol.
    """
    apparatus_list: Optional[List[str]] = Field(
        None, description="List of apparatus or labware involved in this action."
    )
    instrument_list: Optional[List[str]] = Field(
        None, description="List of instruments used in this action."
    )
    location: Optional[Union[str, List]] = Field(
        None, description="Location where the action takes place (e.g., plate position)."
    )
    duration: Optional[float] = Field(
        None, description="Duration of the action in minutes."
    )
    temperature: Optional[str] = Field(
        None, description="Temperature setting for this step, if applicable."
    )

class ActionStep(BaseModel):
    """
    Represents a single detailed action step in the customized protocol.
    """
    step_number: int = Field(
        ..., description="Sequence number of the action step."
    )
    action_type: BioAllowableActionTypes = Field(
        ..., description="Type of laboratory action."
    )
    description: str = Field(
        ..., description="Detailed description of the action to be performed."
    )
    parameters: Optional[StepParameters] = Field(
        None, description="Specific parameters for this action (e.g., volume, temperature)."
    )
    safety_instructions: str = Field(
        ..., description="Safety instructions and precautions for this action step."
    )
    expected_outcome: str = Field(
        ..., description="Expected outcome or result of this action step."
    )

    @field_validator('action_type', mode='before')
    def convert_to_enum(cls, value):
        """Convert the input string to the appropriate enum."""
        if isinstance(value, str):
            return BioAllowableActionTypes[value]
        raise ValueError(f"Value {value} is not a valid {BioAllowableActionTypes.__name__}")

    class Config:
        use_enum_values = False  # Ensure enum is serialized as string by default.
        json_encoders = {
            BioAllowableActionTypes: lambda v: str(v)
        }

class ImageActionFrame(BaseModel):
    """Represents the analysis of a single laboratory image frame."""
    detected_action_type: BioAllowableActionTypes = Field(
        ..., description="The type of laboratory action detected in the image"
    )
    confidence_score: float = Field(
        ..., description="Confidence score for the detected action (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    action_description: str = Field(
        ..., description="Detailed description of the laboratory action being performed in the image"
    )
    
    detected_apparatus: List[str] = Field(
        default=[], description="List of laboratory apparatus detected in the image"
    )
    
    detected_instruments: List[str] = Field(
        default=[], description="List of laboratory instruments detected in the image"
    )
    
    detected_materials: List[str] = Field(
        default=[], description="List of materials or substances detected in the image"
    )   
    
    estimated_step_progress: Optional[float] = Field(
        None, 
        description="Estimated progress of the current action step (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    spatial_information: Optional[Dict[str, Any]] = Field(
        None, description="Spatial information about detected objects (e.g., positions, distances)"
    )
    class Config:
        json_schema_extra = {
            "example": {
                "detected_action_type": "PIPETTING",
                "confidence_score": 0.95,
                "action_description": "Laboratory technician performing precise pipetting of clear liquid into a microplate well",
                "detected_apparatus": ["micropipette", "microplate", "reagent tube"],
                "detected_instruments": ["electronic pipette", "microplate holder"],
                "detected_materials": ["clear liquid solution", "empty wells"],
                "estimated_step_progress": 0.75,
                "spatial_information": {
                    "pipette_position": "above well A1",
                    "working_distance": "appropriate"
                }
            }
        }

class VideoAnalysisRequest(BaseModel):
    """
    Request model for video analysis endpoints.
    """
    video_path: str = Field(..., description="Path to the video file to analyze")
    user_descs: List[str] = Field(..., description="List of questions to analyze the video against")
    model_type: str = Field(default="clip", description="Type of model to use (clip or openai)")
    fps: float = Field(default=2.0, description="Target frames per second for extraction")
    record_top_k_frames: int = Field(default=20, description="Number of top frames to record")
    generate_report: bool = Field(default=True, description="Whether to generate analysis reports")

class VideoAnalysisResponse(BaseModel):
    """
    Response model for video analysis results.
    """
    local_markdown: str = Field(..., description="Path to local markdown report")
    local_html: str = Field(..., description="Path to local HTML report")
    local_pdf: str = Field(..., description="Path to local PDF report")
    s3_markdown: str = Field(..., description="Path to S3 markdown report")
    s3_html: str = Field(..., description="Path to S3 HTML report")
    s3_pdf: str = Field(..., description="Path to S3 PDF report")
    key_frames: dict = Field(..., description="Dictionary mapping questions to key frame S3 URLs")
    frame_descriptions: Dict[str, Optional[str]] = Field(..., description="Dictionary mapping questions to frame descriptions")
    result_dir: str = Field(..., description="Directory containing all results")
    frames_dir: str = Field(..., description="Directory containing extracted frames")
    
class SummarizeRequest(BaseModel):
    video_path: str = Field(..., description="Path or URL to the input video file")
    fps: float = Field(2.0, description="Frames per second to extract from the video")
    config_path: str = Field("config.yaml", description="Path to the configuration file")
    cache_db_path: str = Field("embeddings_cache.db", description="Path to the embeddings cache database")

class SummarizeResponse(BaseModel):
    title: str
    summary: str
