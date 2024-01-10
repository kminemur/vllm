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
# llm = LLM(model="/root/workspace/tiny-llama", device="xpu", dtype="bfloat16")
# llm = LLM(model="/root/workspace/vicuna-7b-v1.5", device="xpu", dtype="float16")
llm = LLM(model="/root/workspace/opt-125m", device="xpu", dtype="float16")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
