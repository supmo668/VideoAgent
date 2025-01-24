"""
Usage:
# Installing latest llava-next: pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# Installing latest sglang.

# Endpoint Service CLI:
python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --port=30000

python3 llama3_llava_main.py --help
"""

import asyncio
import copy
import json
import click
import aiohttp
import requests
from transformers import AutoTokenizer
from llava.conversation import conv_llava_llama_3, SeparatorStyle


def init_conversation(prompt_text):
    # Initialize the conversation with plain style to avoid tokenizer requirement
    conv = copy.deepcopy(conv_llava_llama_3)
    conv.sep_style = SeparatorStyle.PLAIN
    conv.append_message(role=conv.roles[0], message=prompt_text)
    conv.append_message(role=conv.roles[1], message=None)
    return conv


async def send_request(url, data, delay=0):
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            output = await resp.json()
    return output


async def test_concurrent(host, port, prompt):
    url = f"{host}:{port}"

    default_prompt = "<image>\nPlease generate caption towards this image."
    prompt_text = prompt if prompt else default_prompt
    
    # Use the new conversation initialization
    conv_template = init_conversation(prompt_text)
    prompt_with_template = conv_template.get_prompt()
    
    response = []
    for i in range(1):
        response.append(
            send_request(
                url + "/generate",
                {
                    "text": prompt_with_template,
                    "image_data": "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg",
                    "sampling_params": {
                        "max_new_tokens": 1024,
                        "temperature": 0,
                        "top_p": 1.0,
                        "presence_penalty": 2,
                        "frequency_penalty": 2,
                        "stop": "<|eot_id|>",
                    },
                },
            )
        )

    rets = await asyncio.gather(*response)
    for ret in rets:
        print(ret["text"])


def test_streaming(host, port, prompt):
    url = f"{host}:{port}"
    default_prompt = "<image>\nPlease generate caption towards this image."
    prompt_text = prompt if prompt else default_prompt
    
    # Use the new conversation initialization
    conv_template = init_conversation(prompt_text)
    prompt_with_template = conv_template.get_prompt()
    
    pload = {
        "text": prompt_with_template,
        "sampling_params": {
            "max_new_tokens": 1024,
            "temperature": 0,
            "top_p": 1.0,
            "presence_penalty": 2,
            "frequency_penalty": 2,
            "stop": "<|eot_id|>",
        },
        "image_data": "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg",
        "stream": True,
    }
    response = requests.post(
        url + "/generate",
        json=pload,
        stream=True,
    )

    prev = 0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            output = data["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)
    print("")


@click.command()
@click.option('--host', default='http://127.0.0.1', help='Host address for the server')
@click.option('--port', default=30000, help='Port number for the server')
@click.option('--prompt', default=None, help='Custom prompt for the model. If not provided, default caption prompt will be used.')
def main(host, port, prompt):
    """LLaVA-NeXT image captioning tool.
    
    This tool connects to a LLaVA-NeXT server and generates captions for images using both
    concurrent and streaming approaches.
    """
    asyncio.run(test_concurrent(host, port, prompt))
    test_streaming(host, port, prompt)


if __name__ == "__main__":
    main()