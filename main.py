import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from langchain import PydanticParser
from openai import OpenAI
import yaml

from utils.utils import parse_json, parse_text_find_number, parse_text_find_confidence
from utils_clip import frame_retrieval_seg_ego
from utils_general import get_from_cache, save_to_cache
from .models import AnswerFormat, ConfidenceFormat

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
answer_parser = PydanticParser(model=AnswerFormat)
confidence_parser = PydanticParser(model=ConfidenceFormat)


def get_llm_response(
    system_prompt, prompt, json_format=True, model=config["model"]
):
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

    for _ in range(3):
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


def generate_final_answer(question, caption, num_frames):
    prompt = config['prompts']['final_answer'].format(num_frames=num_frames, caption=caption, question=question, AnswerFormat=AnswerFormat.schema_json())
    system_prompt = config['system_prompt']
    response = get_llm_response(system_prompt, prompt, json_format=True)
    parsed_response = answer_parser.parse(response)
    return parsed_response


def generate_description_step(question, caption, num_frames, segment_des):
    prompt = config['prompts']['description_step'].format(num_frames=num_frames, caption=caption, question=question, segment_des=segment_des)
    system_prompt = config['system_prompt']
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def self_eval(previous_prompt, answer):
    prompt = config['prompts']['self_eval'].format(previous_prompt=previous_prompt, answer=answer, ConfidenceFormat=ConfidenceFormat.schema_json())
    system_prompt = config['system_prompt']
    response = get_llm_response(system_prompt, prompt, json_format=True)
    parsed_response = confidence_parser.parse(response)
    return parsed_response


def ask_gpt_caption(question, caption, num_frames):
    prompt = config['prompts']['ask_gpt_caption'].format(num_frames=num_frames, caption=caption, question=question, AnswerFormat=AnswerFormat.schema_json())
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


def run_one_question(video_id, ann, caps, logs):
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    num_frames = len(caps)

    ### Step 1 ###
    sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
    sampled_caps = read_caption(caps, sample_idx)
    previous_prompt, answer_str = ask_gpt_caption(
        formatted_question, sampled_caps, num_frames
    )
    answer = parse_text_find_number(answer_str)
    confidence_str = self_eval(previous_prompt, answer_str)
    confidence = parse_text_find_confidence(confidence_str)
    # segment description
    segment_des = {
        i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
        for i in range(len(sample_idx) - 1)
    }
    ### Step 2 ###
    if confidence < 3:
        try:
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 2: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))

            sampled_caps = read_caption(caps, sample_idx)
            previous_prompt, answer_str = ask_gpt_caption_step(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
            confidence_str = self_eval(previous_prompt, answer_str)
            confidence = parse_text_find_confidence(confidence_str)
        except Exception as e:
            logger.error(f"Step 2 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)

    ### Step 3 ###
    if confidence < 3:
        try:
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            logger.info(f"Step 3: {frame_idx}")
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))
            sampled_caps = read_caption(caps, sample_idx)
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
        except Exception as e:
            logger.error(f"Step 3 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            answer = parse_text_find_number(answer_str)
    if answer == -1:
        logger.info("Answer Index Not Found!")
        answer = random.randint(0, 4)

    logger.info(f"Finished video: {video_id}/{answer}/{ann['truth']}")

    label = int(ann["truth"])
    corr = int(label == answer)
    count_frame = len(sample_idx)

    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": corr,
        "count_frame": count_frame,
    }


def main():
    # if running full set, change subset to fullset
    input_ann_file = "subset_anno.json"
    all_cap_file = "lavila_subset.json"
    json_file_name = "egoschema_subset.json"

    anns = json.load(open(input_ann_file, "r"))
    all_caps = json.load(open(all_cap_file, "r"))
    logs = {}

    tasks = [
        (video_id, anns[video_id], all_caps[video_id], logs)
        for video_id in list(anns.keys())
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(lambda p: run_one_question(*p), tasks)

    json.dump(logs, open(json_file_name, "w"))


if __name__ == "__main__":
    main()
