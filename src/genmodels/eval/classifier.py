# Repository file: src/genmodels/eval/classifier.py
"""CIFAR-10 classifier (ResNet-18, CIFAR variant), trained from scratch, used only as a measuring
instrument for "thematic consistency": how recognisable generated images are, and whether every
class is represented. Typical accuracy after 30 epochs: ~94-95%.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, in_ch = [], 64
        for out_ch, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers += [BasicBlock(in_ch, out_ch, stride), BasicBlock(out_ch, out_ch, 1)]
            in_ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x_uint8):
        x = (x_uint8.float() / 255.0 - MEAN.to(x_uint8.device)) / STD.to(x_uint8.device)
        h = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(h, 1).flatten(1))


def _augment(x, generator):
    """Random crop with 4-pixel zero padding + horizontal flip, per image, on the GPU."""
    N = len(x)
    xp = F.pad(x, (4, 4, 4, 4))
    dy = torch.randint(0, 9, (N,), generator=generator, device=x.device)
    dx = torch.randint(0, 9, (N,), generator=generator, device=x.device)
    rows = (dy[:, None] + torch.arange(32, device=x.device))[:, :, None]
    cols = (dx[:, None] + torch.arange(32, device=x.device))[:, None, :]
    out = xp.permute(0, 2, 3, 1)[torch.arange(N, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(N, generator=generator, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def train_classifier(train_x, train_y, test_x, test_y, epochs=30, batch_size=256, lr=0.1, device="cuda", log=print):
    torch.manual_seed(0)
    model = ResNet18().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    steps_per_epoch = len(train_x) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps_per_epoch, pct_start=0.25)
    use_bf16 = device != "cpu" and torch.cuda.get_device_capability()[0] >= 8
    train_x, train_y = train_x.to(device), train_y.to(device)
    g = torch.Generator(device).manual_seed(0)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x), generator=g, device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                loss = F.cross_entropy(model(_augment(train_x[idx], g)), train_y[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = accuracy(model, test_x, test_y, device)
        log(f"epoch {epoch + 1}/{epochs}: loss {loss.item():.3f}, test acc {acc:.4f} ({time.time() - t0:.0f}s)")
    return model, acc


@torch.no_grad()
def predict_proba(model, x_uint8, device="cuda", batch_size=1000):
    model.eval()
    out = [F.softmax(model(x_uint8[i:i + batch_size].to(device)).float(), dim=1).cpu()
           for i in range(0, len(x_uint8), batch_size)]
    return torch.cat(out)


def accuracy(model, x, y, device="cuda"):
    return (predict_proba(model, x, device).argmax(1) == y.cpu()).float().mean().item()


def load_classifier(path, device="cuda"):
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.eval()
