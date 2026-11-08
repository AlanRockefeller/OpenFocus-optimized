import sys, types, subprocess, json
sys.path.insert(0,'/home/alan/development/OpenFocus')
import cv2, numpy as np
from core.multi_focus_fusion import MultiFocusFusion

def baseline(path, name):
    module=types.ModuleType(name)
    module.__file__='/home/alan/development/OpenFocus/'+path
    exec(compile(subprocess.check_output(['git','show','HEAD:'+path]),module.__file__,'exec'),module.__dict__)
    return module
ai=baseline('fusion_methods/stackmffv3.py','baseline_ai')
old=baseline('core/multi_focus_fusion.py','baseline_mff')
old._stackmffv3_impl=ai._stackmffv3_impl
resolver=old.resolve_runner
def resolve(path):
    prefix='core.multi_focus_fusion:MultiFocusFusion.'
    return getattr(old.MultiFocusFusion,path[len(prefix):]) if path.startswith(prefix) else resolver(path)
old.resolve_runner=resolve
frames=[cv2.imread('/tmp/openfocus-v3-demo/prepared/%04d.tif'%i,cv2.IMREAD_UNCHANGED) for i in range(1,7)]
results={}
for mode in ('soft','hard'):
    for tiled in (False,True):
        outputs=[]
        for cls in (old.MultiFocusFusion,MultiFocusFusion):
            f=cls('stackmffv3',use_gpu=False,exposure_match=False,fusion_parameters={'mode':mode},tile_enabled=tiled,tile_block_size=128,tile_overlap=32,tile_threshold=1)
            outputs.append(f.fuse(frames,thread_count=2))
        results[f'{mode}-{tiled}']={'identical':bool(np.array_equal(*outputs)), 'max_difference':int(np.max(np.abs(outputs[0].astype(float)-outputs[1])))}
print(json.dumps(results,indent=2))
open('/tmp/openfocus-v3-regression.json','w').write(json.dumps(results,indent=2)+'\n')
assert all(r['identical'] for r in results.values())
