#!/usr/bin/env python3

import argparse
import sys
import os
from openai import OpenAI
import easyocr
import warnings
from prompt_toolkit import prompt

def ocr(image_input):
    reader = easyocr.Reader(['ch_sim', 'en'])
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                message=".*'pin_memory' argument is set as true but not supported on MPS now.*",
                                category=UserWarning)

            results = reader.readtext(image_input)
            texts = [result[1] for result in results]
            return '\n'.join(texts).strip()
    except Exception as e:
        print(f"OCR 错误: {str(e)}", file=sys.stderr)
        return None

def get_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Failed to find API key.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def call_deepseek(messages, model, stream):
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream
    )
    return response

def print_response(response, model, stream):
    print(f"🐴🐴🐴({model}) 回答：")
    full_response= ''
    if stream:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                print(content_chunk, end="", flush=True)
                full_response += content_chunk
    else:
        print(response.choices[0].message.content)
        full_response = response.choices[0].message.content
    return full_response


def query_mode(query, contents, image=False, deep=False, stream=True):
    if contents != "":
        if image:
            prompt = f"这是图片 ocr 后的文本：\n\n{contents}\n\n问题:{query}"
        else:
            prompt = f"根据以下内容回答问题：\n\n{contents}\n\n问题:{query}"
    else:
        prompt = query

    professional_prompt=f"我在处理一个任务。如果你是一位专业人士，有更好的方法和建议吗？尽可能全面。任务是："
    prompt = (f"{professional_prompt}{prompt}")
    print(f"🐮🐮🐮 提问：")
    print(f"{prompt}\n")

    model = "deepseek-chat"
    if deep:
        model = "deepseek-reasoner"

    messages = [
        {"role": "user", "content": prompt}
    ]
    response = call_deepseek(messages, model, stream)
    print_response(response, model, stream)

def chat_mode(deep=False, stream=True):
    system_prompt = (
        "你是一位经验丰富的专业助手，请提供全面、深入的建议和最佳实践，"
        "包括但不限于：替代方案、潜在风险、优化策略和行业最佳实践。"
    )

    model = "deepseek-chat"
    if deep:
        model = "deepseek-reasoner"

    messages = [{"role": "system", "content": system_prompt}]
    print("Entering chat mode. Type 'exit', 'quit', or 'q' to exit.")
    print("--------------------------------------")
    while True:
        try:
            print("\n🐮🐮🐮 提问：", flush=True)
            user_input = prompt("").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                break

            messages.append({"role": "user", "content": user_input})
            response = call_deepseek(messages, model, stream)

            reply = print_response(response, model, stream)
            messages.append({"role": "assistant", "content": reply})
        except KeyboardInterrupt:
            print("\nType 'exit', 'quit', or 'q' to exit.")
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="(LLM) AGent for work.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-q', '--query',  nargs='?', const='',
                       help='Query content from standard input.')
    parser.add_argument('-i', '--image', action='store_true',
                       help='Process image from standard input (perform OCR).')
    parser.add_argument('-d', '--deep', action='store_true',
                       help='Use deepseek-reasoner model with chain-of-thought reasoning.')
    parser.add_argument('-c', '--chat', action='store_true',
                       help='Enter interactive chat mode.')
    parser.add_argument('-s', '--stream',
                       help='Enable streaming response output (default: enabled).', default=True)

    args = parser.parse_args()

    if args.chat:
        chat_mode(args.deep, args.stream)
        return

    contents = ''
    if not sys.stdin.isatty():
        if args.image:
            contents = sys.stdin.buffer.read()
            contents = ocr(contents)
            # print(contents)
        else:
            contents = sys.stdin.read().strip()

    if args.query:
        query = args.query
        if not query:
            print("Error: Please provide a query.", file=sys.stderr)
            sys.exit(1)
        query_mode(query, contents, args.image, args.deep, args.stream)


if __name__ == "__main__":
    main()
