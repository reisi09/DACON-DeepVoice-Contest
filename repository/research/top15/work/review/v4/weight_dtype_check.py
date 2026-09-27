"""Question: are the two largest S3 zip members (DF-Arena 1B 4.59 GB, MERT-330M 1.26 GB; byte-identical local copies,
same sizes as `unzip -l`) stored in fp32? If so, fp16 storage would free ~2.9 GB of the 10 GB zip limit, lifting the
1.71 GB headroom that blocks a second large encoder. mmap load: no tensor is materialised. CPU, seconds."""
import collections, sys, torch
for p in sys.argv[1:]:
    sd = torch.load(p, map_location='cpu', mmap=True, weights_only=True)
    sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    n, b = collections.Counter(), collections.Counter()
    for v in sd.values():
        if torch.is_tensor(v):
            n[str(v.dtype)] += v.numel(); b[str(v.dtype)] += v.numel() * v.element_size()
    print(p, {k: f'{n[k]/1e6:.1f}M params, {b[k]/1e9:.3f} GB' for k in n}, 'fp16 bytes would be', f'{sum(n.values())*2/1e9:.3f} GB')
