"""SuperRes — ESRGAN-based 4x Image Super-Resolution"""
import torch, torch.nn as nn, numpy as np
from PIL import Image
import argparse, time

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.c1 = nn.Conv2d(nf,       gc,       3, padding=1)
        self.c2 = nn.Conv2d(nf+gc,    gc,       3, padding=1)
        self.c3 = nn.Conv2d(nf+gc*2,  gc,       3, padding=1)
        self.c4 = nn.Conv2d(nf+gc*3,  gc,       3, padding=1)
        self.c5 = nn.Conv2d(nf+gc*4,  nf,       3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x1=self.lrelu(self.c1(x)); x2=self.lrelu(self.c2(torch.cat([x,x1],1)))
        x3=self.lrelu(self.c3(torch.cat([x,x1,x2],1))); x4=self.lrelu(self.c4(torch.cat([x,x1,x2,x3],1)))
        return self.c5(torch.cat([x,x1,x2,x3,x4],1)) * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64): super().__init__(); self.rdb1=ResidualDenseBlock(nf); self.rdb2=ResidualDenseBlock(nf); self.rdb3=ResidualDenseBlock(nf)
    def forward(self, x): return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64, nb=23, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, nf, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, padding=1)
        self.upsampling = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.2))
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.2), nn.Upsample(scale_factor=2), nn.Conv2d(nf, nf, 3, padding=1), nn.LeakyReLU(0.2))
        self.conv_last = nn.Conv2d(nf, out_ch, 3, padding=1)
    def forward(self, x):
        fea = self.conv_first(x); body = self.conv_body(self.body(fea))
        fea = fea + body; fea = self.upsample(fea)
        return self.conv_last(self.upsampling(fea))

@torch.inference_mode()
def upscale(model, img_path: str, output_path: str, device="cuda"):
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    out = model(tensor).clamp(0,1).squeeze().cpu().numpy().transpose(1,2,0)
    print(f"Upscaled in {(time.perf_counter()-t0)*1000:.1f}ms")
    Image.fromarray((out*255).astype(np.uint8)).save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--output", default="output_4x.png"); p.add_argument("--weights")
    a = p.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RRDBNet().to(device); model.eval()
    if a.weights: model.load_state_dict(torch.load(a.weights, map_location=device))
    if a.input: upscale(model, a.input, a.output, device)
    else: print(f"RRDBNet ready. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
