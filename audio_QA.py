import torch
import gradio as gr
from transformers import (
    AutomaticSpeechRecognitionPipeline,BitsAndBytesConfig,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
from ai_knowledge import ai_knowledge

model_name_or_path = "/llm_model/whisper-large-v2"
language = "chinese"
language_abbr = "zh-CN"
task = "transcribe"

peft_model_path = f"/llm_model/whisper-large-v2-asr-qlora-int8"


# peft_config = PeftConfig.from_pretrained(peft_model_path)
# print(peft_config)

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

# QLoRA 量化配置
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])

model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path, load_in_8bit=True, device_map="auto"
)


model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path,
                                  quantization_config=q_config,
                                  device_map='auto',
                                  trust_remote_code=True)

model = PeftModel.from_pretrained(model, peft_model_path)
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path, language=language, task=task
)
processor = WhisperProcessor.from_pretrained(
    model_name_or_path, language=language, task=task
)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
pipe = AutomaticSpeechRecognitionPipeline(
    model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
)

def transcribe(audio):
    with torch.cuda.amp.autocast():
        text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)[
            "text"
        ]
    return text

# transcribe('test_zh.flac')
def chat_with_pdf(query):

    ai_konw = ai_knowledge(None,query,"运维管理")
    results = ai_konw.chat_with_file()
    return results


audio = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="PEFT LoRA + INT8 Whisper Large V3 Urdu",
    description="Realtime demo for Urdu speech recognition using `PEFT-LoRA+INT8` fine-tuned Whisper Large V3 model.",
)

search = gr.Interface(
    fn = chat_with_pdf,
    inputs="text", outputs=gr.Textbox(label="translate")

)

total = gr.Series(audio,search)

total.launch(server_name="0.0.0.0",ssl_verify=False, ssl_certfile="cert.pem", ssl_keyfile="key.pem")