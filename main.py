import os
# Prevent tinygrad from mis-parsing user env vars like PYTHON="/usr/..." as integers.
# tinygrad expects device flags (e.g. PYTHON, METAL) to be '0' or '1'. If a user
# has PYTHON set to a path, tinygrad's getenv() will try to cast it to int and
# raise. Normalize common device env vars to '0' when their value isn't '0' or '1'.
# Also map deprecated device flags like METAL=1 to DEV=METAL.
for _k in ("PYTHON","CUDA","NV","AMD","CL","QCOM","WEBGPU","CPU","DSP"):
    if _k in os.environ and os.environ[_k] not in ("0","1"):
        os.environ[_k] = "0"
# If any device flag is explicitly enabled, set DEV accordingly for tinygrad.
for _d in ("METAL","CUDA","NV","AMD","CL","QCOM","WEBGPU","CPU","DSP","PYTHON"):
    if os.environ.get(_d) == "1":
        os.environ["DEV"] = _d
        break

import requests
import chess
import chess.pgn
from stockfish import Stockfish
from metal_ai import CharTokenizer, MetalCharLM, build_dataset
import numpy
import random
import io
import time
import pickle
import subprocess
import sys
import re
import json
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
SEQ = 128
MAX_BATCHES = 5
STEPS = 10000
BATCH_SIZE = 256  # REDUCED from 625 to prevent OOM (33MB vs 733MB per batch)
REPEAT_DATASET = 1
os.environ["DEV"] = "METAL"
def load_grammar_texts(path="grammar.txt"):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]
def load_stackoverflow_first_page(tags=None, pagesize=10, max_pages=100):
    if tags is None:
        tags = ["python", "c", "swift", "objective-c", "nim"]

    texts = []
    first_page = True   # <<< GIỮ TRẠNG THÁI

    for tag in tags:
        if first_page:
            page = 1
            first_page = False
        else:
            page = random.randint(1, max_pages)

        url = "https://api.stackexchange.com/2.3/questions"
        params = {
            "pagesize": pagesize,
            "page": page,
            "order": "desc",
            "sort": "activity",
            "site": "stackoverflow",
            "filter": "withbody",
            "tagged": tag
        }

        try:
            js = requests.get(url, params=params, timeout=10).json()
            for it in js.get("items", []):
                body = re.sub(r'<[^>]+>', ' ', it.get("body", ""))
                body = re.sub(r'\s+', ' ', body).strip()
                if len(body) > 50:
                    texts.append(body)
        except:
            pass

        time.sleep(1)

    return texts
def load_stockfish():
    p=subprocess.Popen(["/usr/local/bin/stockfish"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def cmd(c): p.stdin.write(c+"\n"); p.stdin.flush()
    def wait(t):
        for l in p.stdout:
            if t in l: return
    cmd("uci"); wait("uciok"); cmd("isready"); wait("readyok")
    texts=[]
    for _ in range(40):
        b=chess.Board()
        for _ in range(10):
            if b.is_game_over(): break
            b.push(random.choice(list(b.legal_moves)))
        cmd(f"position fen {b.fen()}"); cmd("eval")
        for l in p.stdout:
            if "Final evaluation" in l:
                texts.append(f"FEN: {b.fen()}\nEVAL: {l.strip()}"); break
    cmd("quit"); p.terminate()
    return texts
def load_pgn_txt(path="million_games.pgn"):
    if not os.path.exists(path): return []
    texts = []
    with open(path, "r") as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                # skip malformed game and continue
                continue

            if game is None:
                break

            # Use a Board to convert Move -> SAN safely and handle bad moves
            try:
                board = game.board()
                moves = []
                for move in game.mainline_moves():
                    try:
                        san = board.san(move)
                    except Exception:
                        # fallback to UCI or str(move) if SAN conversion fails
                        try:
                            san = move.uci()
                        except Exception:
                            san = str(move)
                    moves.append(san)
                    try:
                        board.push(move)
                    except Exception:
                        # illegal move encountered, skip this game
                        moves = []
                        break

                if moves:
                    texts.append(" ".join(moves))
            except Exception:
                # if anything unexpected happens, skip this game
                continue
    return texts
def load_english_dict(path="english_words.txt"):
    if os.path.exists(path):
        return [l.strip() for l in open(path,"r",encoding="utf-8") if l.strip()]
    try:
        r=requests.get("https://raw.githubusercontent.com/dwyl/english-words/master/words.txt",timeout=10)
        return [w for w in r.text.splitlines() if w]
    except:
        return []
def load_dolly(path="databricks-dolly-15k.jsonl"):
    if not os.path.exists(path):
        return []
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ins = obj.get("instruction", "").strip()
            ctx = obj.get("context", "").strip()
            resp = obj.get("response", "").strip() or obj.get("output", "").strip()
            if not ins and not resp:
                continue
            if ctx:
                texts.append(ins + "\n" + ctx + "\n" + resp)
            else:
                texts.append(ins + "\n" + resp)
    return texts

def load_wikipedia_html_tech_history(max_pages=10):
    topics = [
        "History_of_computing_hardware",
        "Operating_system",
        "Unix",
        "Linux",
        "Artificial_intelligence"
    ]

    random.shuffle(topics)
    n_pages = random.randint(1, min(max_pages, len(topics)))

    texts = []
    for t in topics[:n_pages]:
        try:
            r = requests.get(f"https://en.wikipedia.org/wiki/{t}", timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            ps = soup.find("div", class_="mw-parser-output").find_all("p")
            txt = " ".join(
                re.sub(r'\[\d+\]', '', p.get_text(" ", strip=True))
                for p in ps if len(p.get_text()) > 60
            )
            if len(txt) > 200:
                texts.append(txt)
        except:
            pass

        time.sleep(1)

    return texts

def load_all_texts():
    """Load tất cả texts từ các nguồn"""
    texts = []
    texts += load_stackoverflow_first_page()
    texts += load_stockfish()
    texts += load_pgn_txt()
    texts += load_english_dict()
    texts += load_dolly()
    texts += load_wikipedia_html_tech_history()
    texts += load_grammar_texts()
    return texts

def build_full_dataset():
    """Load tất cả texts, tạo tokenizer"""
    texts = load_all_texts()
    print("TOTAL TEXTS:", len(texts))
    tokenizer = CharTokenizer(texts)
    random.shuffle(texts)
    return tokenizer, texts

def batch_generator(texts, tokenizer, seq_len, batch_texts_size=50):
    """Generator yield (X, Y) batch by batch"""
    for i in range(0, len(texts), batch_texts_size):
        chunk_texts = texts[i:i+batch_texts_size]
        X, Y = build_dataset(chunk_texts, tokenizer, seq_len=seq_len)
        if len(X) > 0:
            print(f">>> Batch: {len(X)} samples")
            yield numpy.array(X, dtype=numpy.int64), numpy.array(Y, dtype=numpy.int64)

def main():
    tokenizer, texts = build_full_dataset()
    tokenizer.save("tokenizer.json")
    
    model = MetalCharLM(vocab=tokenizer.vocab_size)
    
    # Train streaming: chạy từ generator
    model.train_streaming(
        model,
        batch_generator(texts, tokenizer, seq_len=SEQ),
        steps=STEPS,
        BATCH_SIZE=BATCH_SIZE,
        lr_base=3e-3
    )
    
    model.save("finetune.bybyai")

if __name__=="__main__":
    main()
