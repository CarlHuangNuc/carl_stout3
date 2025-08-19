import base64
import json
import asyncio
import numpy as np
import os, sys, io
import threading
import time
import aiofiles
import librosa
import soundfile
import wave
from typing import Dict, List, Any, Optional
import argparse
import logging
#import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor,  TextStreamer
from qwen_omni_utils import process_mm_info

import uvicorn
from fastapi import FastAPI, Header, Query, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

cur_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(cur_path))
import time

import soundfile as sf
import torch

minicpmo_model_path= "/mnt/huangke1/LIM/Qwen2.5-Omni-7B"

minicpmo_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(minicpmo_model_path, attn_implementation='sdpa', torch_dtype="auto",  device_map="auto")

processor = Qwen2_5OmniProcessor.from_pretrained(minicpmo_model_path)


def extract_audio_by_timestamp(input_file, output_file, start_time, end_time):
    try:
        with wave.open(input_file, 'rb') as wav_in:
            n_channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            frame_rate = wav_in.getframerate()
            n_frames = wav_in.getnframes()

            total_duration = n_frames / float(frame_rate)
            if start_time < 0:
                start_time = 0
            if end_time > total_duration:
                end_time = total_duration
            if start_time >= end_time:
                print("错误: 开始时间必须小于结束时间")
                return False


            start_frame = int(start_time * frame_rate)
            end_frame = int(end_time * frame_rate)
            frames_to_extract = end_frame - start_frame
                                                
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)


            with wave.open(output_file, 'wb') as wav_out:
                wav_out.setnchannels(n_channels)
                wav_out.setsampwidth(sample_width)
                wav_out.setframerate(frame_rate)
                wav_out.setnframes(frames_to_extract)
                wav_in.setpos(start_frame)
                data = wav_in.readframes(frames_to_extract)
                wav_out.writeframes(data)

            print(f"成功提取音频片段: {input_file} -> {output_file}")
            print(f"时间范围: {start_time:.3f}秒 到 {end_time:.3f}秒")
            print(f"片段时长: {end_time - start_time:.3f}秒")
            print(f"声道数: {n_channels}")
            print(f"采样率: {frame_rate}Hz")
            print(f"位深度: {sample_width*8}位")
                                                                                    
            return True

    except Exception as e:
        print(f"错误: 发生未知错误 - {e}")
        return False



input_audio_path = "./test_case1_audio.wav"
test = extract_audio_by_timestamp(input_audio_path,"./test_out.wav",0,3.0)

print(test)
input_audio_path = "./test_out.wav"
conversation = [{"role": "system",
                                        "content": [         {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                                                   ],
                                        },
                                        {"role": "user",                        
                                        "content": [        
                                            {"type": "audio", "audio": input_audio_path},
                                          #  {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
                                            ],
                                        }
                                    ]
                        

USE_AUDIO_IN_VIDEO = True
start_time_4 = time.perf_counter()
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(minicpmo_model.device).to(minicpmo_model.dtype)
start_time = time.perf_counter()

streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10.0)

audios = minicpmo_model.generate(**inputs, streamer=streamer, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=True)
print(type(audios))
count = 0
for wav in audios:
    print(type(wav))
    sf.write("./carl_test_output_"+str(count)+".wav",wav.reshape(-1).detach().cpu().numpy(),samplerate=24000,)
    count = count + 1
end_time = time.perf_counter()
execution_time = end_time - start_time_4
print(f"执行时间: {execution_time:.6f} 秒")
