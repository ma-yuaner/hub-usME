"""字符级自回归语言模型作业：Embedding + 位置向量 + 因果Transformer + Linear。

不是加载预训练大模型：从随机参数学习预测下一个字符。不承诺小语料能生成通顺句子。
依赖：torch。可复用 ../语种识别12类作业/.venv/bin/python 。

先执行：python transformer_lm.py --self-check
训练：python transformer_lm.py --corpus data/语料.txt --epochs 10
续写：python transformer_lm.py --generate --prompt 这家酒店 --new-tokens 100
完整路径或通配符均可传给 --corpus；默认读取脚本旁data/*.txt。
训练记录写入checkpoint旁的.history.json；仅按验证损失选择最佳模型。

关键：x=原文[:-1], y=原文[1:]；每个输入位置都预测下一字符。
因果mask的True表示禁止参考：屏蔽主对角线上方，允许当前位置及之前。
TransformerEncoder虽然名字叫Encoder，但加因果mask后可作为这个简化自回归
模型的主干；本作业没有交叉注意力。用可学习位置Embedding是教学选择。
单向RNN/LSTM无需额外因果mask；双向RNN同样会泄露后文，不能一概说RNN无此问题。
实际实验记录（用户在M1 Pro、32GB内存、MPS上亲手运行，2026-11）：
1. 因果检查：有mask前缀差异0.00000000，无mask为0.13172573。
   标签错位、backward不更新/step更新检查通过。
2. 试训：去重语料抽取20000行，333775字符；dim64/heads4/layers2/ff128，
   seq_len64/batch32；词表3679，参数545759，训练4693块、验证521块。
   epoch train_loss val_loss val_ppl
   1     7.2444     6.4783   650.84
   2     6.3742     6.3326   562.64
   3     6.2191     6.1873   486.52
   续写“今天”仍以零散字词和换行为主。
3. 全量：原语料6021645字符，395286非空行，精确去重149057条重复行，
   剩246229行、4123177字符。连续90/10划分，训练3710859字符、验证412318字符；
   划分后非空整行精确交集为0（不代表无近重复）。词表仅训练部分建立：5244；
   参数747644，训练57982块，验证6442块，验证未知字符0.02%。结构同试训。
   epoch train_loss val_loss val_ppl
   1     5.8758     5.4049   222.50
   2     5.0396     5.0603   157.63
   3     4.7999     4.8978   134.00
   4     4.6679     4.7946   120.86
   5     4.5774     4.7203   112.20
   最佳checkpoint：runs/full_v1/best.pt（第5轮）。未运行独立测试集。
   试训与全量的词表、验证数据不同，不能直接比较两者PPL。
4. 相同prompt“如何学习”、top_k10、new_tokens60、默认seed42：
   temperature0.5样例：
   如何学习速
   怎么做？
   什么牌子的电脑怎么样？
   有什么好看的电影
   怎么样的电脑
   有哪些好看的歌曲
   这个是什么意思？
   这是什么意思？
   temperature0.8样例：
   如何学习速
   怎么做？
   什么生肖
   这个女儿子的网名字
   我的近义词是什么
   我想的电影是什么
   如何提示的小说
   求一个国人的人，不知道这
   此为各一次定性观察，不能据此断言0.5普遍更优；未做多种子生成评估。
5. 陈述句实验statements_v2：15808行、1052234字符，训练947010字符，
   验证105224字符；词表4281，参数1509305，dim128/heads4/layers2/ff512，
   seq_len96/batch32。第12轮train_loss4.7267、val_loss4.8695、val_ppl130.26。
   加载其权重、新建AdamW继续30轮（不是恢复原优化器），仍保持相同语料和结构。
   第40轮train_loss4.2199、val_loss4.6521、val_ppl104.80；
   第41轮train_loss4.2097、val_loss4.6486、val_ppl104.44；
   第42轮train_loss4.1996、val_loss4.6488、val_ppl104.46。
   用户日志及保存规则表明最佳为第41轮，文件runs/statements_v2_long/best.pt。
   固定prompt今天/我们应该/在这个城市，temperature0.6、top_k10、生成80字：
   仍有“我们在我们的人们”等搭配错误、重复和语义断裂，未达到稳定通顺生成。
   验证损失改善不能直接等同自由生成质量改善；尚未进行独立测试集评估。
结论：完成字符级自回归训练和续写，损失持续下降，能产生部分通顺短问句，
但存在词组错误、重复、换行后话题跳转，尚不能稳定生成连贯段落或回答问题。
语料包含大量独立短问句；降低采样温度不等于提高模型学到的能力。
"""
import argparse
import glob
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent


class Blocks(Dataset):
    def __init__(self, ids, length):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.length = length

    def __len__(self):
        return max(0, (len(self.ids) - 1) // self.length)

    def __getitem__(self, i):
        start = i * self.length
        # 连续块，不在切块后随机分训练/验证，避免重叠窗口泄露。
        return (self.ids[start:start+self.length],
                self.ids[start+1:start+self.length+1])


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
        return self.classifier(self.norm(h))  # [批量, 长度, 词表大小]


def epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * y.numel()
            count += y.numel()
    if not count:
        raise ValueError('语料不足一个完整序列，请增加语料或减小seq-len')
    return total / count


@torch.no_grad()
def generate(model, vocab, prompt, n, temperature, top_k, device):
    if not prompt:
        raise ValueError('prompt不能为空')
    model.eval()  # 关闭Dropout；生成时没有backward/step。
    index = {c: i for i, c in enumerate(vocab)}
    ids = [index.get(c, 0) for c in prompt]
    result = prompt
    for _ in range(n):
        x = torch.tensor([ids[-model.seq_len:]], device=device)
        scores = model(x)[0, -1].clone()
        scores[0] = float('-inf')  # 不生成<UNK>占位符。
        if temperature == 0:
            nxt = scores.argmax().item()
        else:
            scores /= temperature
            if top_k:
                threshold = scores.topk(min(top_k, len(vocab)-1)).values[-1]
                scores[scores < threshold] = float('-inf')
            nxt = torch.multinomial(scores.softmax(-1), 1).item()
        ids.append(nxt)
        result += vocab[nxt]
    return result


def self_check():
    torch.manual_seed(42)
    model = LanguageModel(16, seq_len=8, dim=16, heads=2, layers=2, ff=32, dropout=0)
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 9, 10, 11]])
    model.eval()
    with torch.no_grad():
        masked_gap = (model(a)[:, :3] - model(b)[:, :3]).abs().max().item()
        unmasked_gap = (model(a, False)[:, :3] - model(b, False)[:, :3]).abs().max().item()
    assert masked_gap < 1e-5, '有mask时后文不应影响前文'
    assert unmasked_gap > 1e-5, '无mask对照应能观察到后文影响'
    x, y = Blocks(list(range(9)), 4)[0]
    assert x.tolist() == [0, 1, 2, 3] and y.tolist() == [1, 2, 3, 4]
    before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = nn.functional.cross_entropy(model(a).flatten(0, 1), torch.tensor([2, 3, 4, 5, 6, 7]))
    loss.backward()
    assert torch.equal(before, model.classifier.weight)
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    print(f'因果性检查通过：有mask前缀差异={masked_gap:.8f}，无mask={unmasked_gap:.8f}')
    print('标签错位、输出形状、backward不更新/step更新检查通过。此为正确性检查，不是训练成绩。')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', default=str(ROOT / 'data' / '*.txt'))
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'best_transformer.pt')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--ff', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--generate', action='store_true')
    p.add_argument('--self-check', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--init-from', type=Path, help='加载已有权重继续训练；重建AdamW，不是精确恢复旧优化器')
    p.add_argument('--patience', type=int, default=0, help='验证损失连续多少轮未改善就停止，0关闭')
    p.add_argument('--prompt', default='这家酒店')
    p.add_argument('--new-tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.8, help='0为贪心，否则按概率采样')
    p.add_argument('--top-k', type=int, default=20)
    args = p.parse_args()
    if min(args.threads, args.seq_len, args.batch_size, args.dim, args.heads, args.layers, args.ff, args.epochs) < 1:
        p.error('长度、模型尺寸、线程数、批量和轮数必须为正')
    if not 0 < args.val_ratio < 1 or not 0 <= args.dropout < 1 or args.lr <= 0:
        p.error('检查val-ratio、dropout和lr范围')
    if args.temperature < 0 or args.top_k < 0 or args.new_tokens < 0:
        p.error('生成参数不能为负')
    if args.patience < 0:
        p.error('patience不能为负')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.self_check:
        self_check()
        return
    device = torch.device(args.device)
    if args.generate:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = LanguageModel(**ckpt['config']).to(device)
        model.load_state_dict(ckpt['model'])
        print(generate(model, ckpt['vocab'], args.prompt, args.new_tokens,
                       args.temperature, args.top_k, device))
        return
    if args.checkpoint.exists() and not args.overwrite:
        p.error('checkpoint已存在；换--checkpoint或显式使用--overwrite')
    files = sorted(glob.glob(args.corpus))
    if not files:
        p.error('未找到语料；请通过--corpus指定txt路径（通配符加引号）')
    text = '\n'.join(Path(f).read_text(encoding='utf-8-sig') for f in files)
    # 保留原文连续性，先按连续字符分割，再分别切块；词表仅用训练部分。
    split = int(len(text) * (1 - args.val_ratio))
    train_text, val_text = text[:split], text[split:]
    vocab = ['<UNK>'] + sorted(set(train_text))
    if len(vocab) < 3:
        p.error('训练语料字符种类过少')
    index = {c: i for i, c in enumerate(vocab)}
    encode = lambda t: [index.get(c, 0) for c in t]
    tr, va = Blocks(encode(train_text), args.seq_len), Blocks(encode(val_text), args.seq_len)
    if not len(tr) or not len(va):
        p.error('训练/验证至少各需seq-len+1个字符，请增加语料或减小seq-len')
    config = dict(vocab_size=len(vocab), seq_len=args.seq_len, dim=args.dim,
                  heads=args.heads, layers=args.layers, ff=args.ff, dropout=args.dropout)
    initial = None
    if args.init_from:
        if args.init_from.resolve() == args.checkpoint.resolve():
            p.error('继续训练请使用新checkpoint路径，保留原实验')
        initial = torch.load(args.init_from, map_location='cpu', weights_only=True)
        if initial['vocab'] != vocab or initial['config'] != config:
            p.error('继续训练的词表和模型配置必须与原模型完全一致，请核对dim/seq-len等参数')
        old_args = initial['args']
        old_files = sorted(glob.glob(old_args['corpus']))
        old_text = '\n'.join(Path(f).read_text(encoding='utf-8-sig') for f in old_files)
        if old_text != text or old_args['val_ratio'] != args.val_ratio:
            p.error('继续训练须使用相同语料和验证划分')
    model = LanguageModel(**config).to(device)
    if initial:
        model.load_state_dict(initial['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)
    train_loader = DataLoader(tr, args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(va, args.batch_size, shuffle=False, drop_last=False)
    print(f'device={device} train字符={len(train_text)} val字符={len(val_text)} 词表={len(vocab)}')
    print(f'训练块={len(tr)} 验证块={len(va)} 参数={sum(t.numel() for t in model.parameters())}')
    print(f'验证未知字符比例={sum(c not in index for c in val_text)/len(val_text):.2%}')
    print('按连续文本划分，不能保证消除原语料重复；重复内容请先清理。不使用测试集选模型。')
    history, best, stale = [], float('inf'), 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = initial['epoch'] if initial else 0
    if initial:
        best = epoch(model, val_loader, device)
        # 先保存起点：后续不改善时也有可生成的模型，不覆盖旧文件。
        torch.save(dict(initial, val_loss=best), args.checkpoint)
        print(f'加载第{start_epoch}轮权重，起始val_loss={best:.4f}；新AdamW lr={args.lr}，额外训练{args.epochs}轮', flush=True)
    for i in range(1, args.epochs+1):
        train_loss = epoch(model, train_loader, device, optimizer)
        val_loss = epoch(model, val_loader, device)
        if not math.isfinite(train_loss + val_loss):
            raise RuntimeError('损失非有限值，请检查数据/学习率')
        ppl = math.exp(val_loss) if val_loss < 700 else float('inf')
        current_epoch = start_epoch + i
        used_lr = optimizer.param_groups[0]['lr']
        history.append(dict(epoch=current_epoch, train_loss=train_loss, val_loss=val_loss, val_ppl=ppl, lr=used_lr))
        if val_loss < best - 1e-4:
            best = val_loss
            stale = 0
            torch.save(dict(model=model.state_dict(), vocab=vocab, config=config,
                            args=vars(args) | {'checkpoint': str(args.checkpoint),
                                              'init_from': str(args.init_from) if args.init_from else None},
                            epoch=current_epoch, val_loss=val_loss), args.checkpoint)
        else:
            stale += 1
        scheduler.step(val_loss)
        print(f'epoch={current_epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={ppl:.2f} lr={used_lr:.6g}', flush=True)
        args.checkpoint.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if args.patience and stale >= args.patience:
            print(f'验证损失连续{stale}轮未改善，早停；保留验证最好的权重。')
            break
    print(f'最佳模型：{args.checkpoint}；PPL低不等于句子必然通顺，还要检查续写和原文重复。')


if __name__ == '__main__':
    main()
