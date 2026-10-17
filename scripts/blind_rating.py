# Repository file: scripts/blind_rating.py
"""Blind rating of real vs. VAE vs. DDPM images, by a person or by an external vision-language model.

Images from the three sources are shuffled and shown without their source. For each image the rater
answers: realism (1-5), the object class (or 'unrecognisable'), and whether it is a real photo. The
sources are stored separately in key.json and are only used by --score.

Human rater (Colab or local terminal; answers saved after every image, resumable):
    python scripts/blind_rating.py --prepare      # once: draws 50 images per source
    python scripts/blind_rating.py --rate
    python scripts/blind_rating.py --score

External model (e.g. Gemini): the rater only ever receives set.npz-derived images, never key.json:
    python scripts/blind_rating.py --export runs/eval/blind/export      # numbered images + prompt.txt
    # send the sheets and prompt.txt to the model, save its reply as a text file
    python scripts/blind_rating.py --import-answers reply.txt --rater "Gemini"
    python scripts/blind_rating.py --score
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from genmodels.eval.classifier import CLASSES

SOURCES = {"Real images": None, "VAE (conv)": "vae_conv_s0", "DDPM": "ddpm_ancestral1000_s0"}


def prepare(eval_dir, per_source, data_root, seed=0):
    from genmodels.data import load_cifar10
    from genmodels.eval.sampling import load_samples
    rng = np.random.default_rng(seed)
    images, sources, labels = [], [], []
    for name, key in SOURCES.items():
        if key is None:
            x, y = load_cifar10(data_root, train=False)
            idx = rng.choice(len(x), per_source, replace=False)
            images.append(x[idx].numpy())
            labels += y[idx].tolist()
        else:
            x = load_samples(Path(eval_dir) / "samples" / f"{key}.npz").numpy()
            images.append(x[rng.choice(len(x), per_source, replace=False)])
            labels += [-1] * per_source
        sources += [name] * per_source
    order = rng.permutation(len(sources))
    out = Path(eval_dir) / "blind"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "set.npz", images=np.concatenate(images)[order])
    (out / "key.json").write_text(json.dumps({"sources": [sources[i] for i in order], "labels": [labels[i] for i in order]}))
    print(f"prepared {len(order)} images in {out}")


# ---- human rater ---------------------------------------------------------------------------------
def ask(prompt, valid):
    while True:
        a = input(prompt).strip().lower()
        if a in valid:
            return a
        print("  please answer one of:", ", ".join(sorted(valid)))


def rate(eval_dir):
    import matplotlib.pyplot as plt
    d = Path(eval_dir) / "blind"
    images = np.load(d / "set.npz")["images"]
    ans_path = d / "answers.json"
    answers = json.loads(ans_path.read_text()) if ans_path.exists() else {}
    class_menu = "  ".join(f"{i}={c}" for i, c in enumerate(CLASSES))
    for i in range(len(images)):
        if str(i) in answers:
            continue
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(images[i].transpose(1, 2, 0), interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"image {i + 1} / {len(images)}")
        if "inline" in plt.get_backend().lower():  # Jupyter / Colab: render the figure in the output now
            plt.show()
        else:  # local window
            plt.show(block=False)
            plt.pause(0.1)
        realism = ask("realism 1 (clearly fake) .. 5 (photo-realistic): ", {"1", "2", "3", "4", "5"})
        cls = ask(f"class? {class_menu}  -=unrecognisable: ", {str(k) for k in range(10)} | {"-"})
        real = ask("real photo or generated? (r/g): ", {"r", "g"})
        plt.close(fig)
        answers[str(i)] = {"realism": int(realism), "class": None if cls == "-" else int(cls), "real": real == "r"}
        ans_path.write_text(json.dumps(answers))
    print("all images rated; run --score")


# ---- external model rater --------------------------------------------------------------------------
PROMPT = """You are taking part in a blind image-evaluation study.

You will receive {n_sheets} contact sheets with {n} small colour images, numbered 1 to {n}. Each image is 32x32 pixels \
(shown enlarged) and comes from the CIFAR-10 domain: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. \
Some images are real photographs from CIFAR-10 and some were produced by image-generation models. You are not told how \
many of each there are, and the proportions are not necessarily equal, so judge every image on its own.

For EVERY image, give:
1. realism: an integer from 1 to 5
   1 = clearly artificial (heavy blur, smeared shapes, no coherent object)
   2 = mostly artificial, a few plausible elements
   3 = uncertain or mixed
   4 = mostly realistic, minor oddities
   5 = indistinguishable from a real low-resolution photograph
2. class: the CIFAR-10 class the image shows, using exactly one of: airplane, automobile, bird, cat, deer, dog, frog, \
horse, ship, truck. If you cannot recognise any of these objects, write: unrecognisable
3. source: "real" if you believe it is a real photograph, otherwise "generated"

Rules:
- Judge the images independently and only from what you see. Do not try to balance your answers across classes or sources.
- Low resolution alone is not a sign of being generated: every real image is also only 32x32 pixels.
- Rate all {n} images. Do not skip any and do not add explanations.

Output format: plain CSV only, with this header line and then exactly one line per image, in order from 1 to {n}:
id,realism,class,source
1,4,horse,real
2,1,unrecognisable,generated
..."""


def export(eval_dir, out_dir, per_sheet=25, cols=5, scale=4):
    from PIL import Image, ImageDraw, ImageFont
    images = np.load(Path(eval_dir) / "blind" / "set.npz")["images"]  # key.json is deliberately not read here
    out = Path(out_dir)
    (out / "individual").mkdir(parents=True, exist_ok=True)
    n = len(images)
    side, label_h, gap = 32 * scale, 26, 12
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    n_sheets = (n + per_sheet - 1) // per_sheet
    for s in range(n_sheets):
        idx = list(range(s * per_sheet, min(n, (s + 1) * per_sheet)))
        rows = (len(idx) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (side + gap) + gap, rows * (side + label_h + gap) + gap), "white")
        draw = ImageDraw.Draw(sheet)
        for k, i in enumerate(idx):
            r, c = divmod(k, cols)
            x0, y0 = gap + c * (side + gap), gap + r * (side + label_h + gap)
            draw.text((x0, y0), f"#{i + 1}", fill="black", font=font)
            sheet.paste(Image.fromarray(images[i].transpose(1, 2, 0)).resize((side, side), Image.NEAREST), (x0, y0 + label_h))
        sheet.save(out / f"sheet_{s + 1}_images_{idx[0] + 1:03d}-{idx[-1] + 1:03d}.png")
    for i in range(n):
        Image.fromarray(images[i].transpose(1, 2, 0)).resize((side, side), Image.NEAREST).save(out / "individual" / f"image_{i + 1:03d}.png")
    (out / "prompt.txt").write_text(PROMPT.format(n=n, n_sheets=n_sheets), encoding="utf-8")
    print(f"exported {n} images on {n_sheets} sheets to {out} (plus individual/ and prompt.txt)")


def import_answers(eval_dir, reply_path, rater):
    """Parse the model's CSV reply (tolerates code fences, extra text and spacing) into answers.json."""
    d = Path(eval_dir) / "blind"
    n = len(np.load(d / "set.npz")["images"])
    names = {c: k for k, c in enumerate(CLASSES)}
    names.update({"car": 1, "airplane": 0, "aeroplane": 0, "plane": 0})
    answers, problems = {}, []
    for line in Path(reply_path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*\|?\s*#?(\d+)\s*[,;|]\s*(\d)\s*[,;|]\s*([A-Za-z\- ]+?)\s*[,;|]\s*([A-Za-z]+)", line)
        if not m:
            continue
        i, realism, cls, src = int(m.group(1)), int(m.group(2)), m.group(3).strip().lower(), m.group(4).strip().lower()
        if not 1 <= i <= n or not 1 <= realism <= 5:
            problems.append(line.strip())
            continue
        if cls.startswith("unrecog") or cls in ("-", "none"):
            cls_idx = None
        elif cls in names:
            cls_idx = names[cls]
        else:
            problems.append(line.strip())
            continue
        if src not in ("real", "generated", "fake", "synthetic"):
            problems.append(line.strip())
            continue
        answers[str(i - 1)] = {"realism": realism, "class": cls_idx, "real": src == "real"}
    missing = [i + 1 for i in range(n) if str(i) not in answers]
    for p in problems:
        print("could not parse:", p)
    if missing:
        print(f"missing answers for {len(missing)} images: {missing[:30]}{' ...' if len(missing) > 30 else ''}")
    (d / "answers.json").write_text(json.dumps(answers))
    (d / "rater.json").write_text(json.dumps({"rater": rater, "complete": not missing, "n_answers": len(answers)}))
    print(f"imported {len(answers)}/{n} answers from {rater}")


# ---- scoring ---------------------------------------------------------------------------------------
def score(eval_dir):
    d = Path(eval_dir) / "blind"
    key = json.loads((d / "key.json").read_text())
    answers = json.loads((d / "answers.json").read_text())
    rater = json.loads((d / "rater.json").read_text())["rater"] if (d / "rater.json").exists() else "human"
    by_source = {}
    for name in SOURCES:
        idx = [i for i, s in enumerate(key["sources"]) if s == name and str(i) in answers]
        a = [answers[str(i)] for i in idx]
        by_source[name] = {"n": len(a), "realism": float(np.mean([x["realism"] for x in a])),
                           "realism_std": float(np.std([x["realism"] for x in a], ddof=1)) if len(a) > 1 else 0.0,
                           "recognised": float(np.mean([x["class"] is not None for x in a])),
                           "judged_real": float(np.mean([x["real"] for x in a]))}
        if name == "Real images":
            by_source[name]["class_accuracy"] = float(np.mean([x["class"] == key["labels"][i] for i, x in zip(idx, a)]))
    total = sum(v["n"] for v in by_source.values())
    correct = sum(answers[str(i)]["real"] == (s == "Real images") for i, s in enumerate(key["sources"]) if str(i) in answers)
    result = {"rater": rater, "n": total, "real_vs_generated_accuracy": correct / total, "by_source": by_source}
    (d / "score.json").write_text(json.dumps(result, indent=1))
    print(f"rater: {rater} | real-vs-generated accuracy {correct / total:.0%} over {total} images")
    for name, v in by_source.items():
        extra = f" | class accuracy {v['class_accuracy']:.0%}" if "class_accuracy" in v else ""
        print(f"{name:<12} realism {v['realism']:.2f} | named a class {v['recognised']:.0%} | judged real {v['judged_real']:.0%}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="runs/eval")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--rater", default="external model", help="name of the rater, for --import-answers")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--rate", action="store_true")
    g.add_argument("--export", metavar="OUT_DIR")
    g.add_argument("--import-answers", metavar="REPLY_TXT")
    g.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare(args.eval_dir, args.per_source, args.data_root)
    elif args.rate:
        rate(args.eval_dir)
    elif args.export:
        export(args.eval_dir, args.export)
    elif args.import_answers:
        import_answers(args.eval_dir, args.import_answers, args.rater)
    else:
        score(args.eval_dir)


if __name__ == "__main__":
    main()
