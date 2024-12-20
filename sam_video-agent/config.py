############################################################
# Model Configuration
############################################################

SAM_MODEL_NAME = "facebook/sam-vit-huge"

############################################################
# Prompt Templates
############################################################

# Template for creating LLM prompt from SAM summary
LLM_PROMPT_TEMPLATE = """You are given a description of segmented objects from video frames in a laboratory setting.
The segments represent items or areas identified by the Segment Anything Model (SAM). Below is the summary:

{summary}

Based on the above segments, what is the likely laboratory action taking place in this video?"""

############################################################
# Default Parameters
############################################################

DEFAULT_NUM_FRAMES = 5
DEFAULT_TOP_K_SAM_ENTITIES = 3
