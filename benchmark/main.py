import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict

import numpy as np
from langchain_core.output_parsers import PydanticOutputParser
from openai import OpenAI
import yaml
from dotenv import load_dotenv
import click
from utils.utils import parse_json, parse_text_find_number, parse_text_find_confidence
from utils_clip import frame_retrieval_seg_ego
from utils_general import get_from_cache, save_to_cache
from models import AnswerFormat, ConfidenceFormat

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("egoschema_subset.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load configuration from YAML file
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Initialize OpenAI client
client = OpenAI()

# Initialize Pydantic parsers for response validation
answer_parser = PydanticOutputParser(pydantic_object=AnswerFormat)
confidence_parser = PydanticOutputParser(pydantic_object=ConfidenceFormat)


def get_llm_response(
    system_prompt, prompt, json_format=True, 
    model=config["model"],
    n_retry=3
    ) -> str:
    """
    Retrieves response from LLM with caching support.
    
    Uses OpenAI's chat completion API with configurable JSON formatting.
    Implements caching to avoid redundant API calls for identical prompts.
    
    Args:
        system_prompt: Context setting prompt for the LLM
        prompt: Main query or instruction
        json_format: Whether to request JSON-formatted response
        model: LLM model identifier from config
    """
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])
    logger.info(messages)
    cached_value = get_from_cache(key)
    if cached_value is not None:
        logger.info("Cache Hit")
        logger.info(cached_value)
        return cached_value

    print("Not hit cache", key)
    input()

    for _ in range(n_retry):
        try:
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
            response = completion.choices[0].message.content
            logger.info(response)
            save_to_cache(key, response)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"


def generate_final_answer(question, caption, num_frames) -> AnswerFormat:
    """
    Generates final answer using LLM based on accumulated video context.
    
    Formats the question, captions, and frame count into a structured prompt
    for the LLM to make a final decision. Response is validated using
    a Pydantic model (AnswerFormat).
    """
    prompt = config['prompts']['final_answer'].format(
        num_frames=num_frames, caption=caption, question=question, 
        AnswerFormat=AnswerFormat.schema_json()
    )
    system_prompt = config['system_prompt']
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return answer_parser.parse(response)


def generate_description_step(question, caption, num_frames, segment_des) -> dict:
    prompt = config['prompts']['description_step'].format(
        num_frames=num_frames, caption=caption, question=question, segment_des=segment_des
    )
    system_prompt = config['system_prompt']
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def self_eval(previous_prompt, answer):
    """
    Performs self-evaluation of LLM's answer confidence.
    
    Uses LLM to assess its own confidence in the previous answer,
    returning a structured confidence score through ConfidenceFormat.
    Used to determine if additional context gathering is needed.
    """
    prompt = config['prompts']['self_eval'].format(previous_prompt=previous_prompt, answer=answer, ConfidenceFormat=ConfidenceFormat.schema_json())
    system_prompt = config['system_prompt']
    response = get_llm_response(system_prompt, prompt, json_format=True)
    parsed_response = confidence_parser.parse(response)
    return parsed_response


def ask_gpt_caption(question, caption, num_frames):
    prompt = config['prompts']['ask_gpt_caption'].format(
        num_frames=num_frames, caption=caption, 
        question=question, AnswerFormat=AnswerFormat.schema_json()
    )
    system_prompt = config['system_prompt']
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def ask_gpt_caption_step(question, caption, num_frames):
    prompt = config['prompts']['ask_gpt_caption_step'].format(num_frames=num_frames, caption=caption, question=question, AnswerFormat=AnswerFormat.schema_json())
    system_prompt = config['system_prompt']
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def read_caption(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption

def refine_answer(formatted_question: str, sampled_caps: List[str], num_frames: int, 
                 video_id: str, sample_idx: List[int], segment_des: dict) -> Tuple[int, int, List[int]]:
    """
    Refine the answer by analyzing additional frames if confidence is low.
    
    Returns:
        Tuple[int, int, List[int]]: (answer, confidence, updated sample indices)
    """
    try:
        # Get candidate descriptions and frames
        candiate_descriptions = generate_description_step(
            formatted_question, sampled_caps, num_frames, segment_des
        )
        parsed_descriptions = parse_json(candiate_descriptions)
        frame_idx = frame_retrieval_seg_ego(
            parsed_descriptions["frame_descriptions"], video_id, sample_idx
        )
        
        # Update sample indices
        sample_idx = sorted(list(set(sample_idx + frame_idx)))
        sampled_caps = read_caption(captions, sample_idx)
        
        # Get new answer and confidence
        previous_prompt, answer_str = ask_gpt_caption_step(
            formatted_question, sampled_caps, num_frames
        )
        answer = parse_text_find_number(answer_str)
        confidence = parse_text_find_confidence(self_eval(previous_prompt, answer_str))
        
        return answer, confidence, sample_idx
        
    except Exception as e:
        logger.error(f"Refinement Error: {e}")
        answer_str = generate_final_answer(formatted_question, sampled_caps, num_frames)
        return parse_text_find_number(answer_str), 0, sample_idx


def process_captions_and_answer(captions: List[str], questions: List[str], video_id: str) -> List[dict]:
    """
    Process multiple captions and questions to determine answers.
    
    Args:
        captions: List of caption strings describing video frames
        questions: List of questions about the video content
        
    Returns:
        List[dict]: Results for each question
    """
    # Input validation
    assert len(captions) == len(questions), "Number of captions must match number of questions"
    
    results = []
    for caption, question in zip(captions, questions):
        num_frames = len(caption)
        
        # Initial sampling
        sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
        sampled_caps = read_caption(caption, sample_idx)
        
        # Get initial answer and confidence
        previous_prompt, answer_str = ask_gpt_caption(question, sampled_caps, num_frames)
        answer = parse_text_find_number(answer_str)
        confidence = parse_text_find_confidence(self_eval(previous_prompt, answer_str))
        
        # Create segment descriptions
        segment_des = {
            i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
            for i in range(len(sample_idx) - 1)
        }
        
        # Refinement steps if confidence is low
        if confidence < 3:
            answer, confidence, sample_idx = refine_answer(
                question, sampled_caps, num_frames, video_id, sample_idx, segment_des
            )
            
            if confidence < 3:
                answer, _, sample_idx = refine_answer(
                    question, sampled_caps, num_frames, video_id, sample_idx, segment_des
                )
        
        if answer == -1:
            raise ValueError(f"Answer not found for question: {question}")
            
        results.append({
            "answer": answer,
            "count_frame": len(sample_idx)
        })
    
    return results


def run_one_question(video_id: str, annotations: dict, caps: List[str], n_choices: int = 5) -> dict:
    """
    Process a single video question using the annotation format.
    """
    question = annotations["question"]
    answers = [annotations[f"option {i}"] for i in range(n_choices)]
    correct_answer = int(annotations["truth"])
    
    # Format question with answer choices
    formatted_question = (
        f"Here is the question: {question}\nHere are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    
    # Process the question
    results = process_captions_and_answer([caps], [formatted_question])[0]
    results.update({
        "label": correct_answer,
        "corr": int(correct_answer == results["answer"])
    })
    
    # Log results
    json_log_fn = config['files']['json_log_file']
    log_path = f"data/{json_log_fn}"
    
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = {}
    
    logs[video_id] = results
    
    with open(log_path, 'w') as f:
        json.dump(logs, f)
    
    return results


@click.group()
def cli():
    """Video Agent CLI for processing video questions."""
    pass

@cli.command()
@click.option('--input-ann', type=click.Path(exists=True), help='Path to input annotations JSON file')
@click.option('--all-caps', type=click.Path(exists=True), help='Path to all captions JSON file')
def process_files(input_ann, all_caps):
    """Process videos using annotation and caption files."""
    # Load annotations and captions
    anns = json.load(open(input_ann, "r"))
    all_caps = json.load(open(all_caps, "r"))
    
    # Process each video
    tasks = [
        (video_id, anns[video_id], all_caps[video_id])
        for video_id in list(anns.keys())
    ]
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(lambda p: run_one_question(*p), tasks))
    
    # Save results
    json_log_fn = config['files']['json_log_file']
    log_path = f"data/{json_log_fn}"
    
    with open(log_path, 'w') as f:
        json.dump({vid: res for vid, res in zip(anns.keys(), results)}, f)
    
    click.echo(f"Results saved to {log_path}")

@cli.command()
@click.option('--captions', '-c', multiple=True, required=True, help='List of video frame captions')
@click.option('--question', '-q', required=True, help='Question about the video')
@click.option('--options', '-o', multiple=True, required=True, help='Answer options')
@click.option('--correct', '-a', type=int, required=True, help='Index of correct answer')
@click.option('--format-output/--no-format-output', default=True, help='Format output as JSON')
def process_direct(captions, question, options, correct, format_output):
    """Process a single video using direct caption input.
    
    Example:
        python main.py process-direct \
            -c "Person walks" -c "Person sits" \
            -q "What did the person do?" \
            -o "Walk" -o "Run" -o "Sit" \
            -a 0
    """
    # Validate inputs
    if len(options) < 2:
        raise click.BadParameter("At least 2 answer options are required")
    if correct >= len(options):
        raise click.BadParameter("Correct answer index must be less than number of options")
        
    # Format question with options
    formatted_question = (
        f"Here is the question: {question}\nHere are the choices: "
        + " ".join([f"{i}. {opt}" for i, opt in enumerate(options)])
    )
    
    # Process captions and question
    result = process_captions_and_answer(
        captions=[list(captions)],
        questions=[formatted_question]
    )[0]
    
    # Add correctness information
    result.update({
        "label": correct,
        "corr": int(correct == result["answer"])
    })
    
    # Output results
    if format_output:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Answer: {result['answer']}")
        click.echo(f"Correct: {result['corr']}")
        click.echo(f"Frames used: {result['count_frame']}")

@cli.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), required=True, 
              help='Path to JSON file containing captions and questions')
def process_batch(input_file):
    """Process multiple videos from a JSON input file.
    
    Input file format:
    {
        "video_id": {
            "captions": ["cap1", "cap2", ...],
            "question": "What happened?",
            "options": ["opt1", "opt2", ...],
            "correct": 0
        },
        ...
    }
    """
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = {}
    for video_id, video_data in data.items():
        try:
            # Format question with options
            formatted_question = (
                f"Here is the question: {video_data['question']}\nHere are the choices: "
                + " ".join([f"{i}. {opt}" for i, opt in enumerate(video_data['options'])])
            )
            
            # Process video
            result = process_captions_and_answer(
                captions=[video_data['captions']],
                questions=[formatted_question]
            )[0]
            
            # Add correctness information
            result.update({
                "label": video_data['correct'],
                "corr": int(video_data['correct'] == result['answer'])
            })
            
            results[video_id] = result
            
        except Exception as e:
            click.echo(f"Error processing video {video_id}: {str(e)}", err=True)
            results[video_id] = {"error": str(e)}
    
    # Save results
    output_file = input_file.replace('.json', '_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    click.echo(f"Results saved to {output_file}")

if __name__ == "__main__":
    cli()
