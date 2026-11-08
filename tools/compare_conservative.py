"""Compare V3 policies on exactly one prepared stack, Auto Background always off.

python tools/compare_conservative.py --input assets/demo_stack --align --output /tmp/v3-demo --diagnostics
Use --max-size 0 for original resolution; --crop name,x,y,w,h may be repeated.
Crops are 100% pixels of the prepared stack (after optional resizing/alignment).
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.multi_focus_fusion import MultiFocusFusion
from utils.image_utils import quantize_fuse_output
from utils.sorting import natural_sort_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--align', action='store_true', help='align once using OpenFocus; otherwise input must already be aligned')
    parser.add_argument('--max-size', type=int, default=384, help='prepared longest edge; 0 = original size')
    parser.add_argument('--tile-size', type=int, default=0)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--crop', action='append', default=[], metavar='NAME,X,Y,W,H')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted((p for p in args.input.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')), key=lambda p: natural_sort_key(p.name))
    frames = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths]
    if len(frames) < 2 or any(f is None for f in frames):
        parser.error('At least two readable input frames required')
    if args.max_size:
        scale = min(1, args.max_size / max(frames[0].shape[:2]))
        frames = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in frames]
    if args.align:
        from core.registration import _align_openfocus_impl
        frames = _align_openfocus_impl(frames, thread_count=2, equalize_brightness=False, failure_policy='fail')
    if any(f.shape != frames[0].shape for f in frames):
        parser.error('Input shapes differ')
    h, w = frames[0].shape[:2]
    prepared = args.output / 'prepared'
    prepared.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(prepared / f'{i+1:04}.tif'), frame)
    crops = []
    for text in args.crop:
        name, x, y, cw, ch = text.split(',')
        crops.append((name, *map(int, (x, y, cw, ch))))
    if not crops:
        size = min(96, h, w)
        crops = [(f'region-{i+1}', int(x * (w-size)), int(y * (h-size)), size, size)
                 for i, (x, y) in enumerate(((.2,.2),(.5,.3),(.5,.7),(.8,.5),(.1,.8)))]
    if any(x < 0 or y < 0 or cw < 1 or ch < 1 or x+cw > w or y+ch > h for _, x,y,cw,ch in crops):
        parser.error('Crop outside prepared image')
    configs = [('soft', {'mode':'soft'}), ('hard', {'mode':'hard'})]
    for fallback in ('first', 'last'):
        for confidence, advantage in ((.4,.25),(.6,.5),(.8,1.)):
            configs.append((f'conservative-{fallback}-{confidence}', dict(mode='conservative', fallback=fallback, confidence=confidence, sharpness_advantage=advantage)))
    report = dict(policy='subject-preserving-v2', inputs=[str(p) for p in paths], shape=list(frames[0].shape), auto_background=False,
                  exposure_match=False, align=args.align, max_size=args.max_size, tile_size=args.tile_size,
                  crop_coordinates=crops, metrics_note='Fragmentation and edge differences are proxies, not perceptual quality. RSS is cumulative process high-water, including input, ONNX and diagnostics.', runs={})
    # Common edge band on the first processed source, not semantic anatomy.
    base8 = (frames[0] / 257).astype(np.uint8) if frames[0].dtype == np.uint16 else frames[0]
    gray = cv2.cvtColor(base8, cv2.COLOR_BGR2GRAY) if base8.ndim == 3 else base8
    band = cv2.dilate(cv2.Canny(gray, 40, 100), np.ones((7,7),np.uint8)) > 0
    strips = []
    hard = None
    for name, params in configs:
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        maps = {}
        def diagnostics(row, stop, planes, x=0, mask=None):
            for key, plane in planes.items():
                if key not in maps:
                    maps[key] = np.lib.format.open_memmap(folder / f'{key}.npy', mode='w+', dtype=plane.dtype, shape=(h,w))
                target = maps[key][row:stop, x:x+plane.shape[1]]
                np.copyto(target, plane, where=True if mask is None else mask)
        fusion = MultiFocusFusion('stackmffv3', use_gpu=False, exposure_match=False,
            fusion_parameters=params, tile_enabled=bool(args.tile_size),
            tile_block_size=args.tile_size or 1024, tile_threshold=1 if args.tile_size else 2048)
        t0 = time.perf_counter()
        out = fusion.fuse(frames, thread_count=2,
                         **({'diagnostics': diagnostics} if args.diagnostics and params['mode'] == 'conservative' else {}))
        elapsed = time.perf_counter()-t0
        out = quantize_fuse_output(out, any(f.dtype == np.uint16 for f in frames))
        cv2.imwrite(str(folder / 'output.tif'), out)
        record = dict(parameters=params, seconds=elapsed,
                      process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != 'darwin' else 1024**2))
        if name == 'hard':
            hard = out.copy()
        if hard is not None:
            delta = np.abs(out.astype(np.float32)-hard.astype(np.float32))
            record['edge_band_mae_vs_hard_native_units'] = float(delta[band].mean()) if band.any() else None
        if maps:
            accepted = maps['accepted']
            record['override_percent'] = 100 * float(accepted.mean())
            record['fallback_percent'] = 100 - record['override_percent']
            record['override_components'] = int(cv2.connectedComponents(accepted.astype(np.uint8))[0]-1)
            selected = maps['selected']
            record['selection_neighbor_disagreement_percent'] = 100 * float((selected[:,1:] != selected[:,:-1]).mean())
            record['crop_selection_metrics'] = {}
            for label, x, y, cw, ch in crops:
                area = selected[y:y+ch, x:x+cw]
                record['crop_selection_metrics'][label] = {
                    'override_percent': 100 * float(accepted[y:y+ch, x:x+cw].mean()),
                    'horizontal_label_changes': int(np.count_nonzero(area[:,1:] != area[:,:-1])),
                }
            record['fallback_source_number'] = 1 if params['fallback'] == 'first' else len(frames)
            for plane in maps.values():
                plane.flush()
        display = (out / 257).astype(np.uint8) if out.dtype == np.uint16 else out
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(display, (max(1, int(w*min(1,256/w))), max(1,int(h*min(1,256/w)))))
        panels = [('full (thumbnail)', thumb)] + [(label, display[y:y+ch,x:x+cw]) for label,x,y,cw,ch in crops]
        strip = np.full((50+max(p.shape[0] for _,p in panels), sum(p.shape[1]+8 for _,p in panels),3), 35, np.uint8)
        cv2.putText(strip,name,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        offset = 0
        for label,panel in panels:
            cv2.putText(strip,label,(offset+2,39),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
            strip[50:50+panel.shape[0],offset:offset+panel.shape[1]] = panel
            offset += panel.shape[1]+8
        strips.append(strip)
        report['runs'][name] = record
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name, record, flush=True)
    cv2.imwrite(str(args.output/'comparison.jpg'), np.concatenate(strips,axis=0))


if __name__ == '__main__':
    main()
