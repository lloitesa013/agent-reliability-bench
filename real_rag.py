"""
Real RAG layer for the scaled data engine: ONE fixed base LLM (Qwen2.5-Instruct)
+ bge-small retriever.  Kept deliberately thin — the same base model must answer
every system A-E (bench integrity), so all generation goes through here.

Reuses agent.py's prompt shape ("answer ONLY from context, else say I don't know").
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")
_BGE_QPREFIX = "Represent this sentence for searching relevant passages: "

RAG_SYS = ("You are a careful assistant for high-stakes regulated domains (finance, "
           "healthcare, public services). Answer ONLY using the provided context. If the "
           "context does not contain the answer, reply exactly \"I don't know\". Reply with "
           "just the value or a short phrase, no explanation.")
BASE_SYS = ("You are a careful assistant for high-stakes regulated domains. Answer from your "
            "own knowledge. If you are not certain of the specific value, reply exactly "
            "\"I don't know\". Reply with just the value or a short phrase, no explanation.")


class LLM:
    def __init__(self, model_name=BASE_MODEL, device="cuda:0"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def chat_batch(self, sys_prompt, user_prompts, max_new_tokens=48, batch_size=32):
        outs = []
        for i in range(0, len(user_prompts), batch_size):
            chunk = user_prompts[i:i + batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True)
                for u in chunk
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
            for j in range(len(chunk)):
                new = gen[j][enc.input_ids.shape[1]:]
                outs.append(self.tok.decode(new, skip_special_tokens=True).strip())
        return outs


class Retriever:
    def __init__(self, model_name=EMB_MODEL, device="cuda:0"):
        self.emb = SentenceTransformer(model_name, device=device)

    def top_k(self, query, docs, k=1):
        qv = self.emb.encode([_BGE_QPREFIX + query], normalize_embeddings=True)[0]
        dv = self.emb.encode(docs, normalize_embeddings=True)
        sims = dv @ qv
        order = np.argsort(-sims)[:k]
        return [docs[i] for i in order], [float(sims[i]) for i in order]
