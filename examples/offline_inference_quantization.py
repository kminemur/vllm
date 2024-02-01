# To download repo_id model in Hugging Face (only onetime is enough)
from huggingface_hub import snapshot_download
# download_path = snapshot_download(repo_id="huggyllama/llama-7b") # stored in ~/.cache/huggingface/hub/
download_path = snapshot_download(repo_id="Qwen/Qwen-1_8B") # ~/.cache/huggingface/hub/models--Qwen--Qwen-1_8B/snapshots/fa6e214ccbbc6a55235c26ef406355b6bfdf5eed
# may need: pip install tiktoken
# pip instal pip install intel-extension-for-pytorch
print(download_path)

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=10)

# Create an LLM.
# llm = LLM(model="/root/workspace/qwen_18", device="xpu", dtype="float16", disable_log_stats=False,trust_remote_code=True)
llm = LLM(model=download_path, device="xpu", dtype="float16", disable_log_stats=False, trust_remote_code=True, quantization="awq")
# llm = LLM(model="/root/workspace/vicuna-7b-v1.5", device="xpu", dtype="bfloat16")
# llm = LLM(model="/root/workspace/tiny-llama", device="xpu", dtype="float32")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
