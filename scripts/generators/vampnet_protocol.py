"""
VampNet iterative masked-inpainting loop — matches the ACE-Step E2 protocol.

Goal: a SECOND true masked-inpainting model (same operation as ACE-Step), so the
cross-model 'iterative masked editing' claim is apples-to-apples.

Protocol parity with ACE-Step E2:
  * same 10 seeds (outputs/seeds/s01..s10.wav)
  * mask/regenerate the SAME region each iteration (seconds 5-15 of the clip)
  * 8 iterations, feeding previous output back in
  * output naming: outputs_vampnet/editing/sNN_iterKK.wav  (+ sNN_done.json)
  * also a round-trip control (codec encode/decode, no editing) in roundtrip/

!! VERIFY against the current VampNet repo API before trusting the vamp call.
   The masking/inference interface below follows the descripts repo pattern
   (github.com/hugofloresgarcia/vampnet); method names may differ by version.
   The parts to check are marked  ### CHECK ###.

Install (check repo for current instructions):
    pip install vampnet            # or: pip install git+https://github.com/hugofloresgarcia/vampnet
"""
import os, json, numpy as np, soundfile as sf, librosa

WORK = "."                                  # base dir containing outputs/seeds
SEEDS_DIR = "outputs/seeds"
VN_OUT = "outputs_vampnet"
N_ITERS = 8
MASK_START_S = 5.0                          # match ACE-Step repaint region
MASK_END_S   = 15.0
DUR_S = 30                                   # seeds are 30s

for sub in ["editing", "roundtrip"]:
    os.makedirs(f"{VN_OUT}/{sub}", exist_ok=True)


# --------------------------------------------------------------------------
# Load VampNet.  ### CHECK ### — API names per current repo.
# --------------------------------------------------------------------------
def load_vampnet():
    """Returns an interface object. Adjust imports to the installed version."""
    from vampnet.interface import Interface          ### CHECK ###
    interface = Interface.default()                  ### CHECK ### (downloads weights)
    return interface


# --------------------------------------------------------------------------
# One masked-inpainting edit: regenerate [MASK_START_S, MASK_END_S], keep rest.
# --------------------------------------------------------------------------
def vamp_edit(interface, in_wav, out_wav, seed=0):
    import torch
    from vampnet import mask as pmask                ### CHECK ###
    sr = interface.codec.sample_rate                 ### CHECK ###
    y, _ = librosa.load(in_wav, sr=sr, mono=True)
    sig = interface._preprocess(y)                   ### CHECK ### (wrap to AudioSignal)

    z = interface.encode(sig)                        ### CHECK ### tokens [1, n_cb, T]
    T = z.shape[-1]
    # build a time mask: 1 = regenerate (inside region), 0 = keep
    f0 = int((MASK_START_S / DUR_S) * T)
    f1 = int((MASK_END_S   / DUR_S) * T)
    m = torch.zeros_like(z)
    m[:, :, f0:f1] = 1                               # regenerate only the region
    mask = pmask.inpaint(z, m)                       ### CHECK ### or build mask dict directly

    torch.manual_seed(seed)
    zv = interface.vamp(z, mask)                     ### CHECK ### core inpaint call
    out = interface.decode(zv)                       ### CHECK ###
    ya = out.samples.squeeze().cpu().numpy()         ### CHECK ###
    ya = (ya / (np.max(np.abs(ya)) + 1e-8) * 0.95).astype("float32")
    sf.write(out_wav, ya, sr)
    return out_wav


# --------------------------------------------------------------------------
# Round-trip control: encode->decode, NO mask/edit. Pure codec cycling.
# --------------------------------------------------------------------------
def vamp_roundtrip(interface, in_wav, out_wav):
    sr = interface.codec.sample_rate                 ### CHECK ###
    y, _ = librosa.load(in_wav, sr=sr, mono=True)
    sig = interface._preprocess(y)                   ### CHECK ###
    z = interface.encode(sig)                        ### CHECK ###
    out = interface.decode(z)                        ### CHECK ### decode WITHOUT vamping
    ya = out.samples.squeeze().cpu().numpy()
    ya = (ya / (np.max(np.abs(ya)) + 1e-8) * 0.95).astype("float32")
    sf.write(out_wav, ya, sr)
    return out_wav


# --------------------------------------------------------------------------
def run_editing(interface):
    manifest = json.load(open(f"{SEEDS_DIR}/manifest.json"))
    for sid, seed in manifest.items():
        done = f"{VN_OUT}/editing/{sid}_done.json"
        if os.path.exists(done):
            print(f"skip {sid}"); continue
        cur = f"{SEEDS_DIR}/{sid}.wav"
        paths = [cur]
        for i in range(1, N_ITERS + 1):
            op = f"{VN_OUT}/editing/{sid}_iter{i:02d}.wav"
            if os.path.exists(op):
                cur = op; paths.append(op); continue
            try:
                vamp_edit(interface, cur, op, seed=seed.get("seed", 0) + i)
                cur = op; paths.append(op)
                print(f"  vn edit {sid} {i}/{N_ITERS}")
            except Exception as e:
                print(f"  FAIL {sid} iter {i}: {e}"); break
        json.dump(paths, open(done, "w"))
    print("VampNet editing done.")


def run_roundtrip(interface):
    manifest = json.load(open(f"{SEEDS_DIR}/manifest.json"))
    for sid, seed in manifest.items():
        done = f"{VN_OUT}/roundtrip/{sid}_done.json"
        if os.path.exists(done):
            print(f"skip {sid}"); continue
        cur = f"{SEEDS_DIR}/{sid}.wav"
        paths = [cur]
        for i in range(1, N_ITERS + 1):
            op = f"{VN_OUT}/roundtrip/{sid}_iter{i:02d}.wav"
            if os.path.exists(op):
                cur = op; paths.append(op); continue
            try:
                vamp_roundtrip(interface, cur, op)
                cur = op; paths.append(op)
                print(f"  vn rt {sid} {i}/{N_ITERS}")
            except Exception as e:
                print(f"  FAIL {sid} rt {i}: {e}"); break
        json.dump(paths, open(done, "w"))
    print("VampNet roundtrip done.")


if __name__ == "__main__":
    itf = load_vampnet()
    run_editing(itf)
    run_roundtrip(itf)
