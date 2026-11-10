"""第5周字符 Transformer SFT 作业（单文件提交版）。
依赖 Python、PyTorch；不依赖同目录其他 Python 文件。
训练仍需原模型 best.pt、sft_train.jsonl、sft_val.jsonl，不随本源码内嵌。
用法：python sft_submission.py --data-dir 语料目录 --base 原模型.pt --out 新输出目录 --epochs 20
输出：chat.pt、qa.pt、report.json；输出目录必须不存在。

实验：用户亲自运行原版 sft_homework.py，结果位于 runs/sft_my_formal_01/report.json。
本文件仅将相同 LanguageModel 定义内嵌，未重新执行正式训练。
CPU，seed=42，batch_size=32，AdamW，lr=0.001，两模板各20轮。
训练166条：103条源语料改写、63条程序生成。具体数据SHA256随report.json保存。
chat：未见单数字题17/18，原语料改写题2/4，EOS 22/22。
qa：未见单数字题17/18，原语料改写题1/4，EOS 22/22。
chat answer_loss 4.1010 -> 0.1150；qa 4.2837 -> 0.0753。
用户交互观察（chat，贪心，最多8个新词元）：
2、8 -> 8；34、56 -> 56；56、34 -> 56；9、12 -> 19（错误）；均输出EOS。
小样本验证不能证明模板优劣或通用比较能力；验证集不是独立最终测试集。
脚手架由助手提供，训练与上述交互由用户亲手执行；不声称独立编写全部代码。
"""

import argparse
import hashlib
import json
import math
import platform
import random
import re
import unicodedata
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, seq_len=64, dim=128, heads=4,
                 layers=2, ff=256, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError('dim必须能被heads整除')
        self.seq_len = seq_len
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(seq_len, dim)
        # 分别构造层，避免克隆层起始权重相同。
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, ff, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, vocab_size)

    def forward(self, x, causal=True):
        t = x.size(1)
        if not 0 < t <= self.seq_len:
            raise ValueError('输入长度必须在1到seq_len之间')
        h = self.token(x) + self.position(torch.arange(t, device=x.device))
        mask = (torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
                if causal else None)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.classifier(self.norm(h))


EOS = '<EOS>'
IGNORE = -100
TEMPLATES = {'chat': '用户：{question}\n助手：', 'qa': '问题：{question}\n回答：'}
NUMBER_LIST = r'-?\d+(?:(?:\s*[,、]\s*|\s+)-?\d+)+'


def source_examples(path):
    examples = {}
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        row = json.loads(line)
        question = unicodedata.normalize('NFKC', row['instruction'])
        if not re.search(r'最大(?:值|数|的数字)|较大', question):
            continue
        named_pair = re.search(r'数字\s*1\s*:\s*(-?\d+)\s*[,、]?\s*数字\s*2\s*:\s*(-?\d+)\s*[。.]?$', question)
        tail = re.search(rf'(?<![\w.])({NUMBER_LIST})\s*[\])}}。.]?\s*$', question)
        if named_pair:
            values = tuple(map(int, named_pair.groups()))
        elif tail:
            values = tuple(map(int, re.findall(r'-?\d+', tail[1])))
        else:
            continue
        answer_numbers = re.findall(r'(?<![\d.])-?\d+(?!\d|\.\d)', unicodedata.normalize('NFKC', row['output']))
        if str(max(values)) in answer_numbers:
            examples.setdefault(values, f'{path.name}:{line_no}')
    return examples


def make_splits(data_dir):
    train = source_examples(data_dir / 'sft_train.jsonl')
    val = source_examples(data_dir / 'sft_val.jsonl')
    # Keep both orders of each held-out pair out of training.
    held_out = {tuple(sorted((digit, digit % 9 + 1))) for digit in range(1, 10)}
    train = {v: source for v, source in train.items() if tuple(sorted(v)) not in held_out}
    for values in product(range(1, 10), repeat=2):
        if tuple(sorted(values)) not in held_out:
            train.setdefault(values, 'generated/train')
    train_keys = {tuple(sorted(v)) for v in train}
    original_val = {v: source for v, source in val.items() if tuple(sorted(v)) not in train_keys}
    controlled = {v: 'generated/val' for v in product(range(1, 10), repeat=2)
                  if tuple(sorted(v)) in held_out and v not in original_val}
    if not original_val:
        raise ValueError('没有独立的原语料验证样本')
    return train, {'source': original_val, 'single_digit': controlled}


def prompt_for(template, values):
    numbers = '、'.join(map(str, values))
    return template.format(question=f'求最大值：{numbers}')


def make_batches(examples, template, vocab, limit):
    index = {token: i for i, token in enumerate(vocab)}
    rows = []
    for values in examples:
        prompt = [index[c] for c in prompt_for(template, values)]
        answer = [index[c] for c in str(max(values))] + [index[EOS]]
        tokens = prompt + answer
        if len(tokens) - 1 > limit:
            raise ValueError(f'样本超过上下文长度：{values}')
        x = tokens[:-1]
        # The final prompt position predicts the first answer token.
        y = [IGNORE] * (len(prompt) - 1) + answer
        rows.append((x, y))
    return rows


def collate(rows):
    width = max(len(x) for x, _ in rows)
    return (torch.tensor([x + [0] * (width - len(x)) for x, _ in rows]),
            torch.tensor([y + [IGNORE] * (width - len(y)) for _, y in rows]))


def train_epoch(model, batches, optimizer):
    model.train()
    total, count = 0.0, 0
    for x, y in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE)
        if not torch.isfinite(loss):
            raise ValueError('训练损失不是有限数')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = int(y.ne(IGNORE).sum())
        total += loss.item() * n
        count += n
    return total / count


@torch.no_grad()
def evaluate(model, template, vocab, groups):
    model.eval()
    index = {token: i for i, token in enumerate(vocab)}
    rows = []
    for group, examples in groups.items():
        for values, source in examples.items():
            prompt = prompt_for(template, values)
            ids = [index.get(c, 0) for c in prompt]
            answer = []
            stop = 'limit'
            for _ in range(8):
                scores = model(torch.tensor([ids[-model.seq_len:]]))[0, -1]
                scores[0] = -math.inf
                nxt = int(scores.argmax())
                if vocab[nxt] == EOS:
                    stop = 'eos'
                    break
                answer.append(vocab[nxt])
                ids.append(nxt)
            output = ''.join(answer)
            expected = str(max(values))
            rows.append(dict(group=group, source=source, prompt=prompt, values=values,
                             expected=expected, output=output, correct=output.strip() == expected,
                             stop=stop))
    return rows


def summarize(rows):
    return {group: dict(total=len(items), correct=sum(r['correct'] for r in items),
                        eos=sum(r['stop'] == 'eos' for r in items),
                        both=sum(r['correct'] and r['stop'] == 'eos' for r in items))
            for group in sorted({r['group'] for r in rows})
            for items in [[r for r in rows if r['group'] == group]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1 or args.out.exists():
        parser.error('轮次须为正数，输出目录须为新路径')
    seed, batch_size, lr = 42, 32, 0.001
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    random.seed(seed)
    base = torch.load(args.base, map_location='cpu', weights_only=True)
    train, val = make_splits(args.data_dir)
    all_text = ''.join(prompt_for(template, v) + str(max(v))
                       for template in TEMPLATES.values() for v in train)
    vocab = base['vocab'] + sorted(set(all_text) - set(base['vocab'])) + [EOS]
    # Shared expansion: copy every original parameter and keep identical new rows for A/B.
    config = dict(base['config'], vocab_size=len(vocab))
    model = LanguageModel(**config)
    with torch.no_grad():
        for name, old in base['model'].items():
            model.state_dict()[name][:old.shape[0]].copy_(old)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    baseline = LanguageModel(**base['config'])
    baseline.load_state_dict(base['model'])
    files = [args.base, args.data_dir / 'sft_train.jsonl', args.data_dir / 'sft_val.jsonl']
    report = dict(seed=seed, batch_size=batch_size, lr=lr, optimizer='AdamW', epochs=args.epochs,
                  device='cpu', python=platform.python_version(), torch=str(torch.__version__),
                  decoding='greedy, at most 8 new tokens', config=config, templates=TEMPLATES,
                  hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  added_tokens=vocab[len(base['vocab']):], train_size=len(train),
                  train_source=sum(source != 'generated/train' for source in train.values()),
                  train_examples=[dict(values=v, source=source) for v, source in train.items()],
                  results={})
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"训练 {len(train)} 条（原语料 {report['train_source']} 条），验证各组 { {k: len(v) for k, v in val.items()} }", flush=True)
    for name, template in TEMPLATES.items():
        before = evaluate(baseline, template, base['vocab'], val)
        model.load_state_dict(initial)
        torch.manual_seed(seed)
        batches = DataLoader(make_batches(train, template, vocab, model.seq_len),
                             batch_size=batch_size, shuffle=True, collate_fn=collate,
                             generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        history = []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, batches, optimizer)
            history.append(loss)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(f'{name} epoch={epoch:02d} answer_loss={loss:.4f}', flush=True)
        after = evaluate(model, template, vocab, val)
        checkpoint = args.out / f'{name}.pt'
        torch.save(dict(model=model.state_dict(), config=config, vocab=vocab, template=template), checkpoint)
        report['results'][name] = dict(checkpoint=str(checkpoint), history=history, before=before, after=after,
                                       summary_before=summarize(before), summary_after=summarize(after))
        print(f'{name} before={summarize(before)} after={summarize(after)}', flush=True)
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'报告：{args.out / "report.json"}')


if __name__ == '__main__':
    main()
