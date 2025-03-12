> 初衷是英语废物看纯英语课程太费劲， 英文听不懂， 看字幕又觉得跟看文稿一样，于是想给英语课程中文配音。转成了汉语 然后配音换音轨，做好同步，这样看这些英语的课程就不用听英文了。为了防止出现奇怪的翻译问题，字幕是双语的，这样如果音频翻译很奇怪，可以凑合着对着看一下

# Video Transformer

## 功能

该项目是一个视频处理工具，主要功能包括：
1. 从视频中提取音频并分割音频。
2. 使用 Whisper 模型将音频转录为文本。
3. 使用 OpenAI API 将转录文本翻译为中文。
4. 使用 edge_tts 库将翻译后的文本合成为语音。
5. 将合成的语音与原视频合并。
6. 生成双语字幕文件并将字幕添加到视频中。

## 使用方法

### 环境配置

1. 克隆项目到本地：
    ```sh
    git clone https://github.com/zijinjun/videoTransformer.git
    cd videoTransformer
    ```

2. 创建并激活虚拟环境：
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # 对于 Windows 系统，使用 .venv\Scripts\activate
    ```

3. 安装依赖：
    ```sh
    pip install -r requirements.txt
    ```

### 配置文件

在运行代码之前，需要配置 OpenAI API 密钥。可以在 `.env` 文件中设置以下环境变量：
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_KEY_FREE=your_openai_api_key_free
```

### 运行代码

1. 将要处理的视频文件命名为 `input.mp4` 并放置在项目根目录下。
2. 运行 `demo.py`：
    ```sh
    python demo.py
    ```

### 生成的文件

运行代码后，会生成以下文件：
- `audio.wav`：从视频中提取的音频文件。
- `split_parts.txt`：分割后的音频文件列表。
- `transcript.json`：转录后的文本文件。
- `translated.json`：翻译后的文本文件。
- `tts_audio.mp3`：合成的语音文件。
- `output.mp4`：合成语音后的输出视频文件。
- `subtitles.srt`：生成的双语字幕文件。
- `output_with_subtitles.mp4`：添加字幕后的最终视频文件。

## 依赖

项目依赖的库在 `requirements.txt` 文件中列出：
```
asyncio
os
json
subprocess
whisper
pydub
openai
edge_tts
```

## 注意事项

- 确保安装了 FFmpeg，并且 FFmpeg 可执行文件在系统 PATH 中。
- 运行代码时，请确保网络连接正常，以便调用 OpenAI API 进行翻译。

## 未完成

- 为了实现音轨同步，一定程度上会有一些误差，需要进一步优化。
- 一些口语化内容翻译会有翻译不准确的情况，需要进一步优化。
- 代码中可能存在一些潜在的 bug，需要进一步测试和修复。