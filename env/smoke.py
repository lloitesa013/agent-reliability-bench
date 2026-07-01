import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

print("device_count:", torch.cuda.device_count(), "name0:", torch.cuda.get_device_name(0))

MODEL = "Qwen/Qwen2.5-14B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
msgs = [
    {"role": "system", "content": "Answer with ONLY the value, no words."},
    {"role": "user", "content": "Policy: 'Daily ATM withdrawal limit is 1,000 USD.' Question: What is the daily ATM withdrawal limit?"},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to("cuda:0")
out = model.generate(**ids, max_new_tokens=32, do_sample=False)
print("GEN:", repr(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()))

emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda:0")
v = emb.encode(["prime loan APR?", "Prime personal loan APR is 6.9%.", "Subprime personal loan APR is 14.9%."])
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
print("BGE cos(q,prime):", round(cos(v[0], v[1]), 3), " cos(q,subprime):", round(cos(v[0], v[2]), 3))
print("SMOKE_OK")
