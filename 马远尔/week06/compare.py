"""Compare classifiers on one decontaminated ChnSentiCorp dev split."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
import unicodedata
from zipfile import ZipFile

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def load_rows(archive, split):
    data = archive.read(f"ChnSentiCorp/{split}.tsv").decode("utf-8-sig")
    rows = []
    reader = csv.DictReader(io.StringIO(data), delimiter="\t", quoting=csv.QUOTE_NONE)
    for index, row in enumerate(reader):
        if row["label"] not in {"0", "1"} or not row["text_a"].strip():
            raise ValueError(f"{split} row {index} is not a labeled nonblank review")
        rows.append((row.get("qid", str(index)), row["text_a"], int(row["label"])))
    return rows


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def unique_rows(rows, seen):
    result = []
    for row in rows:
        key = normalized(row[1])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def rule(text):
    positive = ("干净", "满意", "推荐", "热情", "不错", "方便", "很棒")
    negative = ("不满意", "不推荐", "不干净", "失望", "很差", "糟糕", "投诉")
    good = sum(term in text for term in positive)
    bad = sum(term in text for term in negative)
    return int(good > bad)


def score(y, predictions, fit_seconds, predict_seconds):
    return {
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "ms_per_review": round(1000 * predict_seconds / len(y), 3),
        "metrics": classification_report(
            y, predictions, labels=[0, 1], target_names=["negative", "positive"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def fit_bert(model_path, train, dev, args):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, local_files_only=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    generator = torch.Generator().manual_seed(args.seed)

    def batch_encode(batch):
        return tokenizer(
            [row[1] for row in batch], padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )

    losses = []
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        indices = torch.randperm(len(train), generator=generator).tolist()
        for offset in range(0, len(indices), args.batch):
            group = [train[i] for i in indices[offset:offset + args.batch]]
            labels = torch.tensor([row[2] for row in group])
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch_encode(group), labels=labels).loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(group)
        losses.append(running_loss / len(train))
        print(f"bert epoch={epoch + 1} mean_train_loss={losses[-1]:.4f}", flush=True)
    training_seconds = perf_counter() - start
    model.eval()
    start = perf_counter()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(dev), args.batch):
            group = dev[offset:offset + args.batch]
            predictions.extend(model(**batch_encode(group)).logits.argmax(dim=1).tolist())
    model_result = score([row[2] for row in dev], predictions, training_seconds, perf_counter() - start)
    model_result.update({
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "epochs": args.epochs,
        "batch": args.batch,
        "threads": args.threads,
        "max_length": args.max_length,
        "learning_rate": 3e-5,
        "train_losses": losses,
        "model_dir": str(model_path),
        "model_revision": args.model_revision,
        "checkpoint_policy": "evaluate the final in-memory state; no weights written",
        "parameter_count": sum(p.numel() for p in model.parameters()),
    })
    return model_result, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a fresh file")
    with ZipFile(args.dataset) as archive:
        train_raw = load_rows(archive, "train")
        dev_raw = load_rows(archive, "dev")
    if (len(train_raw), len(dev_raw)) != (9600, 1200):
        parser.error("expected 9600 train and 1200 dev rows in the PaddleNLP release")
    seen = set()
    train_unique = unique_rows(train_raw, seen)
    dev = unique_rows(dev_raw, seen)
    train = train_unique
    if 0 < args.train_size < len(train):
        train, _ = train_test_split(
            train, train_size=args.train_size, stratify=[row[2] for row in train],
            random_state=args.seed,
        )
    train_x, train_y = [row[1] for row in train], [row[2] for row in train]
    dev_x, dev_y = [row[1] for row in dev], [row[2] for row in dev]
    results = {}
    predictions = {}

    start = perf_counter()
    predictions["keywords"] = [rule(text) for text in dev_x]
    results["keywords"] = score(dev_y, predictions["keywords"], 0, perf_counter() - start)
    for name, classifier in (
        ("tfidf_logreg", LogisticRegression(max_iter=1000, random_state=args.seed)),
        ("tfidf_svm", LinearSVC(random_state=args.seed)),
    ):
        pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, max_features=60000),
            classifier,
        )
        start = perf_counter()
        pipeline.fit(train_x, train_y)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        predictions[name] = pipeline.predict(dev_x).tolist()
        results[name] = score(dev_y, predictions[name], fit_seconds, perf_counter() - start)
    if args.model:
        results["rbt3_finetune"], predictions["rbt3_finetune"] = fit_bert(args.model, train, dev, args)

    examples = []
    for i, (qid, _, label) in enumerate(dev):
        examples.append({"qid": qid, "label": label, **{k: v[i] for k, v in predictions.items()}})
    report = {
        "dataset": "PaddleNLP ChnSentiCorp",
        "archive_md5": hashlib.md5(args.dataset.read_bytes()).hexdigest(),
        "evaluation": "dev, deduplicated against all train and within dev; unlabeled test untouched",
        "seed": args.seed,
        "sklearn": sklearn.__version__,
        "train_raw": len(train_raw),
        "train_unique": len(train_unique),
        "dev_raw": len(dev_raw),
        "train_labels": dict(Counter(train_y)),
        "dev_labels": dict(Counter(dev_y)),
        "train_row_ids": [row[0] for row in train],
        "results": results,
        "predictions": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"train={len(train)} dev={len(dev)} removed_from_dev={len(dev_raw)-len(dev)}")
    for name, metric in results.items():
        print(f"{name}: macro_f1={metric['metrics']['macro avg']['f1-score']:.4f} "
              f"accuracy={metric['metrics']['accuracy']:.4f} fit_s={metric['fit_seconds']:.3f} "
              f"predict_ms_per_review={metric['ms_per_review']:.3f}")
    print(args.out)


if __name__ == "__main__":
    main()




# 第六周分类对照作业

## 实验问题

把课件的 TNEWS 新闻主题分类换成 ChnSentiCorp 中文评论情感二分类。比较关键词、字级 TF-IDF 加逻辑回归、字级 TF-IDF 加线性 SVM，以及三层中文预训练编码器 RBT3 的监督微调。四条路线使用同一评估集，三条需要训练的路线使用相同训练样本。结果适用于本数据与设置。TNEWS 的课件结果使用另一任务和另一评估集，留作课程背景。

## 数据与配置

- 数据来源：[PaddleNLP ChnSentiCorp](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/chnsenticorp.py)，发布包 MD5 为 `7ef61b08ad10fbddf2ba97613f071561`。标签 0 为负评、1 为正评。
- 原始训练 9,600 条，验证 1,200 条。对文本做 NFKC、大小写及空白归一化。训练内部去重后 8,249 条，分层抽 1,000 条，正评 529、负评 471。验证同训练全集交叉去重并在验证集内部去重，剩余 843 条，正评 490、负评 353。未给标签的 test.tsv 未读取。
- 初次冒烟用默认 TSV 引号解析，训练集误读为 9,146 行。 复核原始表格后改用 QUOTE_NONE，并添加训练、验证行数校验，初次数字作废。
- 随机种子 42。传统模型使用同一套字级 1 至 3-gram TF-IDF 设置。词典规则以正负词命中个数做判断，对平局预测负评。预训练模型 [HFL RBT3](https://huggingface.co/hfl/rbt3) 修订 `0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`，权重文件 SHA-256 为 `3e04f7477f55dffce2a2fbc4d0ba35068415162a9e92e3d5cc74a49781ba4eb0`。CPU，AdamW，学习率 3e-5，batch 16，max length 96，2 epoch，在内存中评估末轮权重。
- Python 3.14.6，PyTorch 2.14.1，Transformers 4.57.6，scikit-learn 1.9.1。报告中的训练耗时为单次测量，RBT3 包含本地模型加载，预测耗时包含分词或向量化。

## 重跑

以下操作从 `/Users/mima0000/Documents/个人学习` 执行。数据及权重保存在临时目录，可按同一校验值重新取得。独立环境加载现有 PyTorch，避免改动原语种识别项目依赖。

```bash
LAB=/tmp/wanwu-classification
mkdir -p "$LAB/rbt3"
curl -fsSL https://bj.bcebos.com/paddlenlp/datasets/ChnSentiCorp.zip \
  -o "$LAB/ChnSentiCorp.zip"
md5 "$LAB/ChnSentiCorp.zip"

REV=0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c
for FILE in config.json vocab.txt pytorch_model.bin; do
  curl -fSL "https://huggingface.co/hfl/rbt3/resolve/$REV/$FILE" \
    -o "$LAB/rbt3/$FILE"
done
shasum -a 256 "$LAB/rbt3/pytorch_model.bin"

uv venv --python '语种识别12类作业/.venv/bin/python' "$LAB/env"
uv pip install --python "$LAB/env/bin/python" \
  scikit-learn==1.9.1 transformers==4.57.6
SOURCE_SITE=$('语种识别12类作业/.venv/bin/python' -c \
  'import site; print(site.getsitepackages()[0])')
TARGET_SITE=$("$LAB/env/bin/python" -c \
  'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$SOURCE_SITE" > "$TARGET_SITE/learning-torch.pth"
```

**本次实际实验命令**

```bash
HF_HUB_OFFLINE=1 /tmp/wanwu-classification/env/bin/python -B \
  '第六周分类对照作业/compare.py' \
  /tmp/wanwu-classification/ChnSentiCorp.zip \
  --out '第六周分类对照作业/results-four-way.json' \
  --train-size 1000 --model /tmp/wanwu-classification/rbt3 \
  --model-revision 0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c \
  --epochs 2 --batch 16 --max-length 96 --threads 4
```

复跑时将 --out 改为新文件名。不传 --model 时只跑三条传统路线。输出 JSON 含训练行号、验证 qid、逐条预测、每类指标和混淆矩阵，便于核查同一输入的错分。训练代码和作业证据由助手运行，个人作答与亲手复现待完成。

## 结果

本次实验和结果复核退出码均为 0。原始输出包含 `train=1000 dev=843 removed_from_dev=357`。RBT3 的平均训练损失第 1 轮为 0.5675，第 2 轮为 0.3431。按逐条预测重算四条路线的 Macro-F1，与原报告一致。传统路线的验证 qid、真实标签和预测也与修正数据读取后的独立传统基线运行一致。

| 方法 | Macro-F1 | Accuracy | 拟合或训练秒数 | 预测毫秒/条 |
|---|---:|---:|---:|---:|
| 关键词词典 | 0.6176 | 0.6216 | 0 | 0.002 |
| TF-IDF + 逻辑回归 | 0.8465 | 0.8493 | 0.259 | 0.089 |
| TF-IDF + 线性 SVM | 0.8581 | 0.8600 | 0.241 | 0.089 |
| RBT3 监督微调 | 0.8711 | 0.8731 | 32.245 | 3.432 |

混淆矩阵的行是真实标签，列是预测标签，均按负评、正评排序。SVM 为 `[[313,40],[78,412]]`，RBT3 为 `[[316,37],[70,420]]`。相对 SVM，RBT3 修正 67 条，同时新增 56 条错分，净多答对 11 条。Macro-F1 提高约 1.30 个百分点。在这次 CPU、两种方案各自批处理口径下，RBT3 的每条平均预测耗时约为 SVM 的 39 倍。未测 p95、生产并发或多次耗时波动。

## 错例和下一轮

- qid 129，负评。评论提到发热、厚重、物流慢。SVM 预测正评，RBT3 正确。先检查 SVM 的特征和权重，再判断具体是哪条线索被削弱。
- qid 86，负评。开头认可位置，后面批评设施与服务。SVM 判对，RBT3 误判为正评。可先看分词后哪些语句进入了 96 token 窗口，再做长度对照；当前结果未证明是截断或开头正面措辞所致。

**学习者**　这次 RBT3 最好，能说预训练一定比 SVM 更值吗？

**万物**　这批验证题多答对 11 条，预测耗时也更高。把两套模型都错、各自改对的评论翻出来，才能判断多出来的正确对业务是否重要。现在是单种子、小训练子集的验证结果。下一轮把种子固定，用同一份留出验证集，只改 RBT3 的输入上限 96 → 128，比较同一批错例，并记下耗时变化。零样本提示与 LoRA 未纳入这四路对照。

## 复核

在工作区根目录运行 `/tmp/wanwu-classification/env/bin/python /tmp/wanwu-classification/verify_result.py`，实际输出 `PASS: shared 1000 train / 843 dev, four macro-F1 scores recomputed from predictions`，退出 0。该临时检查逐条交叉核对传统基线和最终记录，未加入提交目录。
