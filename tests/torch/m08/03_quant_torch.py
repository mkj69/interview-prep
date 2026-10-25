import torch

def quantize_per_channel_int8(w):
    # w: (out, in). symmetric absmax per output channel
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.float() * scale

if __name__ == '__main__':
    torch.manual_seed(0)
    w = torch.randn(256, 1024) * torch.randn(256, 1).exp()
    q, s = quantize_per_channel_int8(w)
    err = (w - dequantize(q, s)).norm() / w.norm()
    assert q.dtype == torch.int8 and err < 0.02, err
    # per-element error bounded by half a step
    assert ((w - dequantize(q, s)).abs() <= s / 2 + 1e-6).all()
    print('ok', err.item())
