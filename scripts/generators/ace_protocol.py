#!/usr/bin/env python3
"""
ACE-Step v1.5 — iterative editing (E2) + VAE round-trip control (C1).

LENGTH-LOCKED rewrite. ACE-Step's `repaint` re-encodes the whole src_audio each
call and its VAE does not preserve sample-exact length, so feeding the previous
output back in made clips shrink cumulatively (30 -> 27 -> ... -> 19s). That
moved the 5-15s repaint region across a shifting timeline and invalidated all
fixed-region scoring. Fix: pin every output to exactly TARGET_DUR seconds before
it becomes the next iteration's input, so the region maps to the same seconds at
every k. Mirrors what the SAO latent-clamp protocol guarantees structurally.

One-time setup (shell, only if ACE-Step isn't already installed):
    git clone https://github.com/ace-step/ACE-Step-1.5.git
    pip install -e ACE-Step-1.5/acestep/third_parts/nano-vllm
    grep -v flash-attn ACE-Step-1.5/requirements.txt > /tmp/r.txt && pip install -r /tmp/r.txt
    pip install -e ACE-Step-1.5

Run:
    python ace_protocol.py
Then score with drift_v4 at region=(4,6) against the C1 roundtrip ceiling.
"""
import os, json, shutil, numpy as np, librosa, soundfile as sf

# ============================== CONFIG ==============================
PROJECT_ROOT  = os.environ.get("ACE_ROOT", "/content/ACE-Step-1.5")
HF_HOME       = os.environ.get("HF_HOME", "acestep_protocol/hf_cache")   
SR            = 48000
TARGET_DUR    = 30.0          # every clip is pinned to exactly this length
N_ITERS       = 8
REPAINT_START = 5.0           # repaint seconds 5-15 of the 30s clip
REPAINT_END   = 15.0
INFER_STEPS   = int(os.environ.get("ACE_INFER_STEPS", "8"))   # 8 = turbo; raise (e.g. 30) for cleaner, less-artificial fills
CLEAN_E2_C1   = True          # wipe old E2/C1 clips + done files before regen (KEEPS seeds)
# BASE = the dir that holds outputs/ ; "." makes SEEDS_DIR == "outputs/seeds", i.e. CWD-relative,
# matching what rerank_bestofn.py / sao_inpaint_protocol.py read. Run ACE from the SAME dir you
# run rerank from (e.g. /root) so the new seeds land where rerank looks. Override via ACE_BASE.
BASE = os.environ.get("ACE_BASE", ".")

SEEDS_DIR = f"{BASE}/outputs/seeds"
E2_DIR    = f"{BASE}/outputs/E2_iterative"
C1_DIR    = f"{BASE}/outputs/C1_roundtrip"
for d in (SEEDS_DIR, E2_DIR, C1_DIR, HF_HOME):
    os.makedirs(d, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME

SEEDS = [
    dict(id='s01', caption='calm piano solo in C major',                keyscale='C Major',  bpm=72,  duration=30, seed=101),
    dict(id='s02', caption='upbeat electronic dance in A minor',        keyscale='A Minor',  bpm=128, duration=30, seed=102),
    dict(id='s03', caption='jazz trio with piano bass drums in F major',keyscale='F Major',  bpm=110, duration=30, seed=103),
    dict(id='s04', caption='ambient strings in D major',                keyscale='D Major',  bpm=60,  duration=30, seed=104),
    dict(id='s05', caption='indie folk acoustic guitar in G major',     keyscale='G Major',  bpm=90,  duration=30, seed=105),
    dict(id='s06', caption='cinematic orchestral in E minor',           keyscale='E Minor',  bpm=80,  duration=30, seed=106),
    dict(id='s07', caption='funk bass groove in Bb major',              keyscale='Bb Major', bpm=100, duration=30, seed=107),
    dict(id='s08', caption='minimal techno in C minor',                 keyscale='C Minor',  bpm=135, duration=30, seed=108),
    dict(id='s09', caption='bossa nova guitar in E major',              keyscale='E Major',  bpm=95,  duration=30, seed=109),
    dict(id='s10', caption='slow blues in B minor',                     keyscale='B Minor',  bpm=65,  duration=30, seed=110),
    # --- E-D seed expansion (n=10 -> 20): wider genre/type coverage; ACE-generated to keep the
    #     seed pool homogeneous with s01-s10 (see ablation §7 scaling note). seeds 111-120. ---
    dict(id='s11', caption='driving rock band with electric guitars',   keyscale='E Minor',  bpm=140, duration=30, seed=111),
    dict(id='s12', caption='trap beat with heavy 808 bass and hi-hats',  keyscale='F Minor',  bpm=140, duration=30, seed=112),
    dict(id='s13', caption='reggae groove with offbeat guitar skank',    keyscale='A Major',  bpm=76,  duration=30, seed=113),
    dict(id='s14', caption='heavy metal with distorted guitars and double kick', keyscale='D Minor', bpm=160, duration=30, seed=114),
    dict(id='s15', caption='solo violin classical adagio',               keyscale='G Minor',  bpm=100, duration=30, seed=115),
    dict(id='s16', caption='retro synthwave with analog pads and arpeggios', keyscale='C Major', bpm=118, duration=30, seed=116),
    dict(id='s17', caption='afrobeat with polyrhythmic percussion and horns', keyscale='D Major', bpm=112, duration=30, seed=117),
    dict(id='s18', caption='drum and bass with fast breakbeat and sub bass', keyscale='A Minor', bpm=174, duration=30, seed=118),
    dict(id='s19', caption='soulful gospel with hammond organ and choir', keyscale='Bb Major', bpm=84,  duration=30, seed=119),
    dict(id='s20', caption='lo-fi hip hop with mellow rhodes and vinyl crackle', keyscale='Eb Major', bpm=82, duration=30, seed=120),
    # --- Session-2 A/B expansion (n=20 -> 40, 2026-07-28): the head-to-head needs ~2x the
    #     opposed-ordering pairs to reach power (see doc/mos/human_mos_study_protocol.md §10).
    #     ALL INSTRUMENTAL — no vocals/lyrics, which the low-risk ethics classification
    #     rests on. seeds 121-140. ---
    dict(id='s21', caption='classical string quartet allegro',            keyscale='A Major',  bpm=120, duration=30, seed=121),
    dict(id='s22', caption='flamenco guitar with rapid strumming',        keyscale='A Minor',  bpm=104, duration=30, seed=122),
    dict(id='s23', caption='bluegrass banjo and fiddle breakdown',        keyscale='G Major',  bpm=132, duration=30, seed=123),
    dict(id='s24', caption='disco with four on the floor and string stabs', keyscale='D Minor', bpm=120, duration=30, seed=124),
    dict(id='s25', caption='salsa with brass section and congas',         keyscale='C Minor',  bpm=98,  duration=30, seed=125),
    dict(id='s26', caption='ska with offbeat upstroke guitar and horns',  keyscale='F Major',  bpm=150, duration=30, seed=126),
    dict(id='s27', caption='dubstep with wobble bass and sparse drums',   keyscale='F Minor',  bpm=140, duration=30, seed=127),
    dict(id='s28', caption='baroque harpsichord invention',               keyscale='D Major',  bpm=96,  duration=30, seed=128),
    dict(id='s29', caption='celtic fiddle jig with bodhran',              keyscale='D Major',  bpm=116, duration=30, seed=129),
    dict(id='s30', caption='middle eastern oud with hand percussion',     keyscale='D Minor',  bpm=88,  duration=30, seed=130),
    dict(id='s31', caption='japanese koto traditional melody',            keyscale='E Minor',  bpm=70,  duration=30, seed=131),
    dict(id='s32', caption='solo harp new age arpeggios',                 keyscale='F Major',  bpm=64,  duration=30, seed=132),
    dict(id='s33', caption='big band swing with trumpet section',         keyscale='Bb Major', bpm=180, duration=30, seed=133),
    dict(id='s34', caption='surf rock with reverb drenched guitar',       keyscale='E Major',  bpm=144, duration=30, seed=134),
    dict(id='s35', caption='psychedelic rock with hammond organ',         keyscale='G Minor',  bpm=108, duration=30, seed=135),
    dict(id='s36', caption='downtempo chillout with warm pads',           keyscale='C Major',  bpm=92,  duration=30, seed=136),
    dict(id='s37', caption='piano house groove with filtered chords',     keyscale='A Minor',  bpm=124, duration=30, seed=137),
    dict(id='s38', caption='tango with bandoneon and strings',            keyscale='D Minor',  bpm=116, duration=30, seed=138),
    dict(id='s39', caption='gamelan percussion ensemble',                 keyscale='G Major',  bpm=100, duration=30, seed=139),
    dict(id='s40', caption='brass band march with tuba and snare',        keyscale='Eb Major', bpm=112, duration=30, seed=140),
]

# ============================== ACE INIT ==============================
def init_ace():
    import torch  # noqa: F401
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    dit = AceStepHandler()
    dit.initialize_service(
        project_root=PROJECT_ROOT,
        config_path='acestep-v15-turbo',
        device='cuda',
        offload_to_cpu=False,
        quantization=None,
    )
    llm = LLMHandler()
    print("✓ ACE-Step ready")
    return dit, llm

# ============================== HELPERS ==============================
def pin_length(path, dur=TARGET_DUR, sr=SR):
    """Force a clip to exactly `dur` seconds: trim if long, zero-pad if short.
    This is the length-lock that stops cumulative VAE trim from shifting the
    repaint region across iterations. Padding only ever touches the tail
    (>= real audio end), so scoring inside 5-15s is unaffected."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(round(dur * sr))
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    sf.write(path, y.astype('float32'), sr)

def _gen(dit, llm, params):
    from acestep.inference import GenerationConfig, generate_music
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format='wav')
    return generate_music(dit, llm, params, cfg, save_dir=params._save_dir)

def _repaint_params(seed, src_audio, save_dir, rstart, rend, seed_val):
    """Build a repaint GenerationParams pinned to TARGET_DUR."""
    from acestep.inference import GenerationParams
    p = GenerationParams(
        task_type='repaint',
        src_audio=src_audio,
        caption=seed['caption'], keyscale=seed['keyscale'], bpm=seed['bpm'],
        duration=TARGET_DUR,                 # request 30 (VAE may still trim -> pin_length fixes)
        repainting_start=rstart,
        repainting_end=rend,
        seed=seed_val,
        inference_steps=INFER_STEPS,
        guidance_scale=0.0,
        thinking=False,
    )
    p._save_dir = save_dir
    return p

def repaint_edit(dit, llm, seed_meta, in_wav, out_wav, seed_val, mask_s=(REPAINT_START, REPAINT_END)):
    """E-D candidate generator: ONE native ACE repaint pass of `in_wav` over mask_s -> out_wav.

    Mirrors sao.inpaint_edit's contract so rerank_bestofn.py can swap backends: only `seed_val`
    varies across a seed's N candidates. `seed_meta` must carry caption/keyscale/bpm (the seed's
    own generation params) -- ACE repaints in-distribution, which is why it stays coherent where
    SAO's training-free latent-replacement does not. Raises on failure (rerank prints & continues).
    """
    rstart, rend = mask_s
    save_dir = os.path.dirname(out_wav) or "."
    os.makedirs(save_dir, exist_ok=True)
    p = _repaint_params(seed_meta, in_wav, save_dir, rstart, rend, seed_val)
    r = _gen(dit, llm, p)
    if not r.success:
        raise RuntimeError(f"ACE repaint failed: {r.error}")
    shutil.move(r.audios[0]['path'], out_wav)
    pin_length(out_wav)                              # lock to 30s (same as the drift arms)
    return out_wav

def _clean(dirpath):
    for f in os.listdir(dirpath):
        if f.endswith('.wav') or f.endswith('_done.json'):
            os.remove(os.path.join(dirpath, f))

# ============================== SEEDS ==============================
def generate_seeds(dit, llm):
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    manifest = {}
    for s in SEEDS:
        out_path = f"{SEEDS_DIR}/{s['id']}.wav"
        if os.path.exists(out_path):
            manifest[s['id']] = {**s, 'path': out_path}
            print(f"skip seed {s['id']} (exists)")
            continue
        p = GenerationParams(
            task_type='text2music',
            caption=s['caption'], keyscale=s['keyscale'], bpm=s['bpm'],
            duration=s['duration'], seed=s['seed'],
            inference_steps=INFER_STEPS, use_adg=False,
            guidance_scale=0.0, thinking=False,
        )
        cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format='wav')
        r = generate_music(dit, llm, p, cfg, save_dir=SEEDS_DIR)
        if not r.success:
            print(f"✗ seed {s['id']} failed: {r.error}"); continue
        shutil.move(r.audios[0]['path'], out_path)
        pin_length(out_path)                     # seeds locked to 30s too
        manifest[s['id']] = {**s, 'path': out_path}
        print(f"✓ seed {s['id']}")
    with open(f"{SEEDS_DIR}/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest

# ============================== E2 / C1 ==============================
def run_arm(dit, llm, manifest, dirpath, mode):
    """mode='editing' -> repaint 5-15s each iter (cumulative on previous output).
       mode='roundtrip' -> zero-width repaint at 10s (pure re-encode control)."""
    results = {}
    for seed_id, seed in list(manifest.items())[:10]:
        done_file = f"{dirpath}/{seed_id}_done.json"
        if os.path.exists(done_file):
            with open(done_file) as f: results[seed_id] = json.load(f)
            print(f"skip {mode} {seed_id} (complete)"); continue

        current = seed['path']; paths = [current]; ok = True
        for i in range(1, N_ITERS + 1):
            out_path = f"{dirpath}/{seed_id}_iter{i:02d}.wav"
            if os.path.exists(out_path):
                current = out_path; paths.append(out_path); continue

            if mode == 'editing':
                p = _repaint_params(seed, current, dirpath,
                                    REPAINT_START, REPAINT_END, seed['seed'] + i)
            else:  # roundtrip: zero-width edit, fixed seed (region unchanged)
                p = _repaint_params(seed, current, dirpath, 10.0, 10.0, seed['seed'])

            r = _gen(dit, llm, p)
            if not r.success:
                print(f"  ✗ {mode} {seed_id} iter {i}: {r.error}"); ok = False; break
            shutil.move(r.audios[0]['path'], out_path)
            pin_length(out_path)                 # <-- the fix, every iteration
            current = out_path; paths.append(out_path)
            print(f"  ✓ {mode} {seed_id} iter {i}/{N_ITERS}  "
                  f"dur={librosa.get_duration(path=out_path):.1f}s")

        results[seed_id] = paths
        if ok:
            with open(done_file, 'w') as f: json.dump(paths, f)
    print(f"{mode} done.")
    return results

# ============================== VERIFY ==============================
def verify_durations():
    bad = 0
    for dirpath, tag in ((E2_DIR, 'E2'), (C1_DIR, 'C1')):
        for f in sorted(os.listdir(dirpath)):
            if not f.endswith('.wav'): continue
            d = librosa.get_duration(path=os.path.join(dirpath, f))
            if abs(d - TARGET_DUR) > 0.05:
                print(f"  !! {tag}/{f}  dur={d:.2f}s  (NOT {TARGET_DUR})"); bad += 1
    if bad == 0:
        print(f"✓ all E2/C1 clips are {TARGET_DUR:.0f}s — region is fixed across iterations")
    else:
        print(f"✗ {bad} clips off-length — do not score until fixed")

# ============================== MAIN ==============================
if __name__ == "__main__":
    # ACE_SEEDS_ONLY=1 -> generate/append seeds and STOP (used to grow the E-D seed pool
    # from 10 to 20 without touching the drift arms). generate_seeds is skip-if-exists, so
    # this only makes the NEW seeds and rewrites manifest.json with all of them.
    seeds_only = os.environ.get("ACE_SEEDS_ONLY", "") not in ("", "0", "false", "False")
    if CLEAN_E2_C1 and not seeds_only:
        _clean(E2_DIR); _clean(C1_DIR)
        print("cleaned old E2/C1 clips (seeds preserved)")

    dit, llm = init_ace()
    manifest = generate_seeds(dit, llm)
    if seeds_only:
        print(f"ACE_SEEDS_ONLY: {len(manifest)} seeds in manifest; skipping drift arms.")
        raise SystemExit(0)
    run_arm(dit, llm, manifest, E2_DIR, 'editing')
    run_arm(dit, llm, manifest, C1_DIR, 'roundtrip')
    verify_durations()
