import asyncio
import os
import json
import subprocess

import whisper
from pydub import AudioSegment
from openai import \
    OpenAI  # 引入OpenAI客户端类<button class="citation-flag" data-index="2"><button class="citation-flag" data-index="5">
import edge_tts  # 中文TTS库<button class="citation-flag" data-index="7">


def main():
    # client = OpenAI(
    #     api_key=os.environ['OPENAI_API_KEY'],
    #     base_url='https://api.deerapi.com/v1'
    # )

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key=os.environ['OPENAI_API_KEY_FREE'],
        timeout=30
    )

    input_video = 'input.mp4'
    audio_file = 'audio.wav'
    split_parts_file = 'split_parts.txt'
    transcript_file = 'transcript.json'
    translated_file = 'translated.json'
    tts_audio = 'tts_audio.mp3'
    output_video = 'output.mp4'

    # 音频提取与分割
    extract_audio(input_video, audio_file)
    split_audio(audio_file, split_parts_file)

    # 语音转文字
    if not os.path.exists(transcript_file):
        transcribe_audio(client, split_parts_file, transcript_file)

    # 文本翻译
    translate_text(client, transcript_file, translated_file)

    # 语音合成与视频合并
    asyncio.run(generate_tts(translated_file, tts_audio))
    merge_video(input_video, tts_audio, output_video)

    # 新增字幕处理
    subtitle_file = 'subtitles.srt'
    generate_srt_subtitles(translated_file, subtitle_file)
    final_output_video = 'output_with_subtitles.mp4'
    add_subtitles_to_video(output_video, subtitle_file, final_output_video)

def extract_audio(input_video, output_audio):
    if not os.path.exists(output_audio):
        print("正在提取音频...")
        subprocess.run([
            'ffmpeg', '-i', input_video, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2', output_audio
        ], check=True)
        print("音频提取完成")


def split_audio(input_audio, split_list_file):
    if not os.path.exists(split_list_file):
        print("正在分割音频...")
        audio = AudioSegment.from_wav(input_audio)
        split_duration = 1000 * 60 * 25  # 每25分钟分割

        output_format = "wav"  # 使用MP3格式压缩<button class="citation-flag" data-index="2"><button class="citation-flag" data-index="7">
        bitrate = "32k"  # 低比特率（可调为64k/128k）<button class="citation-flag" data-index="9">
        sample_rate = "22050"  # 降低采样率<button class="citation-flag" data-index="9">
        channels = "1"  # 单声道<button class="citation-flag" data-index="3">

        parts = []
        for i in range(0, len(audio), split_duration):
            start_time = i / 1000  # 转换为秒
            duration = split_duration / 1000
            part_name = f"audio_part_{i // split_duration}.{output_format}"

            # 构建FFmpeg命令以控制编码参数
            cmd = [
                'ffmpeg',
                '-i', input_audio,
                '-ss', str(start_time),
                '-t', str(duration),
                '-acodec', 'libmp3lame',
                '-b:a', bitrate,
                '-ar', sample_rate,
                '-ac', channels,
                '-loglevel', 'error',  # 抑制冗余输出
                part_name
            ]

            try:
                subprocess.run(cmd, check=True)
                parts.append(part_name)
            except subprocess.CalledProcessError as e:
                print(f"音频分割失败：{e}")
                return

        with open(split_list_file, 'w') as f:
            f.write('\n'.join(parts))
        print(f"音频分割完成（格式：{output_format}，比特率：{bitrate}）")

def transcribe_audio_local(model_size, audio_path):
    """使用本地Whisper模型转录音频"""
    # 根据<button class="citation-flag" data-index="3"><button class="citation-flag" data-index="7">加载模型
    model = whisper.load_model(model_size)

    # 转录并返回结果
    result = model.transcribe(audio_path,
                              language="en",  # 自动检测语言
                              fp16=False)  # CPU使用时禁用FP16

    # 提取带时间戳的分段数据
    segments = []
    for seg in result['segments']:
        segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'].strip()
        })
    return segments


def transcribe_audio(client, split_list, output_file):
    if not os.path.exists(output_file):
        segments = []
        total_offset = 0
        with open(split_list, 'r') as f:
            for part in f.read().splitlines():
                base_name = os.path.splitext(os.path.basename(part))[0]
                transcript_file = f"{base_name}_transcript.json"  # <button class="citation-flag" data-index="3"><button class="citation-flag" data-index="5">
                if os.path.exists(transcript_file):  # 检查中间结果文件是否存在<button class="citation-flag" data-index="5">
                    with open(transcript_file, 'r') as f_part:
                        part_segments = json.load(f_part)  # 直接加载已保存的转录结果<button class="citation-flag" data-index="5">
                else:
                    part_segments = transcribe_audio_local("small", part)  # <button class="citation-flag" data-index="3"><button class="citation-flag" data-index="7">
                    with open(transcript_file, 'w') as f_part:  # 保存中间结果供后续使用<button class="citation-flag" data-index="5">
                        json.dump(part_segments, f_part)
                for seg in part_segments:
                    seg['start'] += total_offset
                    seg['end'] += total_offset
                segments.extend(part_segments)
                part_duration = AudioSegment.from_wav(part).duration_seconds
                total_offset += part_duration
        with open(output_file, 'w') as f_out:
            json.dump({"segments": segments}, f_out)
        print("转写完成")


def translate_text(client, input_file, output_file):
    # 检查是否存在已有的翻译结果<button class="citation-flag" data-index="5"><button class="citation-flag" data-index="9">
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            translated_segments = existing_data.get('segments', [])
    else:
        translated_segments = []

    # 创建已翻译的索引集合，根据分段在原始列表中的位置标识<button class="citation-flag" data-index="5">
    translated_indices = {seg['index']: seg for seg in translated_segments}

    with open(input_file, 'r') as f:
        data = json.load(f)
    input_segments = data['segments']

    for idx, seg in enumerate(input_segments):
        seg_index = idx

        if seg_index in translated_indices:
            continue  # 跳过已翻译的分段<button class="citation-flag" data-index="5">

        # 调用API进行翻译<button class="citation-flag" data-index="5"><button class="citation-flag" data-index="7">
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位精通英文的翻译家，你需要把用户提供的文本翻译为中文，只输出翻译后的句子"},
                {"role": "user", "content": "把以下文本翻译为中文，只输出翻译后的句子，文本如下：\n" + seg['text']}
            ],
            temperature=0.3,
            max_tokens=4096  # 控制输出长度<button class="citation-flag" data-index="5">
        )

        translated_seg = seg.copy()
        translated_seg['text'] = response.choices[0].message.content
        translated_seg['index'] = seg_index  # 添加唯一标识符<button class="citation-flag" data-index="5">

        translated_segments.append(translated_seg)

        # 立即保存中间结果到文件，确保断点续传<button class="citation-flag" data-index="5"><button class="citation-flag" data-index="9">
        with open(output_file, 'w') as f_out:
            json.dump({"segments": translated_segments}, f_out)

        print(f"Translate progress {len(translated_segments)}/{len(input_segments)}")

    print("翻译完成")


async def generate_tts(translated_file, output_audio):
    if os.path.exists(output_audio):
        print(f"检测到 {output_audio} 已存在，跳过语音合成阶段")
        return

    with open(translated_file, 'r') as f:
        data = json.load(f)
    segments = data['segments']
    segments.sort(key=lambda x: x['start'])  # 按时间排序分段

    total_duration = AudioSegment.from_wav("audio.wav").duration_seconds * 1000
    full_audio = AudioSegment.silent(duration=int(total_duration))

    for i, seg in enumerate(segments):
        start = int(seg['start'] * 1000)
        text = seg['text']
        temp_path = f"temp_{start}.mp3"

        if os.path.exists(temp_path):
            print(f"复用已存在的 {temp_path}")
            voice_segment = AudioSegment.from_mp3(temp_path)
        else:
            print(f"正在合成分段：{text}（起始时间：{start}ms）")
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            await communicate.save(temp_path)
            voice_segment = AudioSegment.from_mp3(temp_path)

        # 计算允许的最大时长（到下一个分段的开始时间）
        if i < len(segments) - 1:
            next_start = segments[i + 1]['start'] * 1000
            max_duration = next_start - start
        else:
            max_duration = len(voice_segment)  # 最后一个分段不限制

        if len(voice_segment) > max_duration:
            try:
                speed_ratio = len(voice_segment) / max_duration
                # 动态调整分块大小，防止过小的音频无法分割
                chunk_size = min(1000, int(len(voice_segment) / 2))
                voice_segment = voice_segment.speedup(
                    playback_speed=speed_ratio,
                    chunk_size=chunk_size,
                    crossfade=10  # 平滑过渡防止爆音
                )
            except Exception as e:
                # 如果加速失败，直接截断音频
                print(f"加速失败：{e}, 转为截断处理")
                voice_segment = voice_segment[:max_duration]  # 直接截断到允许的最大时长

            if len(voice_segment) > max_duration:
                # 二次保障，确保不超过时间限制
                voice_segment = voice_segment[:max_duration]

        full_audio = full_audio.overlay(voice_segment, position=start)

    full_audio.export(output_audio, format="mp3")
    print("语音合成完成")

    # 清理临时文件
    for seg in segments:
        start = int(seg['start'] * 1000)
        os.remove(f"temp_{start}.mp3")


def merge_video(input_video, input_audio, output_video):
    if not os.path.exists(output_video):
        # 引用<button class="citation-flag" data-index="8">的FFmpeg命令格式，强制替换音频轨并明确映射流
        cmd = [
            'ffmpeg',
            '-i', input_video,
            '-i', input_audio,
            '-c:v', 'copy',          # 复制原视频流<button class="citation-flag" data-index="8"><button class="citation-flag" data-index="9">
            '-c:a', 'aac',           # 使用AAC编码新音频<button class="citation-flag" data-index="2"><button class="citation-flag" data-index="8">
            '-map', '0:v:0',         # 仅使用原视频的视频流<button class="citation-flag" data-index="8"><button class="citation-flag" data-index="9">
            '-map', '1:a:0',         # 仅使用新音频的音频流<button class="citation-flag" data-index="8"><button class="citation-flag" data-index="9">
            '-shortest',             # 以较短的媒体时长结束<button class="citation-flag" data-index="8">
            '-y',                    # 覆盖输出文件
            output_video
        ]
        subprocess.run(cmd, check=True)
        print("视频合并完成")


def generate_srt_subtitles(translated_file, subtitle_file):
    """根据翻译后的JSON生成双语SRT字幕文件（上为英文原文，下为中文翻译）"""
    # 读取翻译后的文件（中文）
    with open(translated_file, 'r') as f_translated:
        translated_data = json.load(f_translated)
    translated_segments = translated_data['segments']

    # 读取原始转录文件（英文）
    with open('transcript.json', 'r') as f_transcript:
        transcript_data = json.load(f_transcript)
    transcript_segments = transcript_data['segments']

    with open(subtitle_file, 'w', encoding='utf-8') as f:
        for idx, (transcript_seg, translated_seg) in enumerate(zip(transcript_segments, translated_segments), 1):
            start = transcript_seg['start']
            end = transcript_seg['end']
            original_text = transcript_seg['text'].strip()
            translated_text = translated_seg['text'].strip()

            # 时间戳格式转换函数（HH:MM:SS,mmm）
            def format_time(sec):
                hours = int(sec // 3600)
                minutes = int((sec % 3600) // 60)
                seconds = int(sec % 60)
                millisecs = int((sec - int(sec)) * 1000)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millisecs:03d}"

            # 双语字幕格式：英文在上，中文在下
            f.write(f"{idx}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{original_text}\n")  # 英文原文
            f.write(f"{translated_text}\n\n")  # 中文翻译（换行分隔）

    print(f"双语字幕已生成至 {subtitle_file}")  # <button class="citation-flag" data-index="4"><button class="citation-flag" data-index="7"><button class="citation-flag" data-index="9">

def add_subtitles_to_video(input_video, subtitle_file, output_video):
    """使用FFmpeg将字幕文件添加到视频中"""
    if not os.path.exists(output_video):
        cmd = [
            'ffmpeg',
            '-i', input_video,
            '-vf', f"subtitles={subtitle_file}",
            '-c:a', 'copy',  # 复用音频流
            '-movflags', '+faststart',  # 优化MP4文件
            output_video
        ]
        subprocess.run(cmd, check=True)
        print(f"字幕已添加到 {output_video}")

if __name__ == '__main__':
    main()