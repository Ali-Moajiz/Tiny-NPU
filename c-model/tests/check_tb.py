"""TB=1 runtime checks using a freshly compiled image, in temporary storage."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.compiler.resolve()))
    from compiler import compile_model, write_hex
    from isa import encode_program
    from verify import oracle, pack_tensor
    import numpy as np
    layers=[dict(weights=[[int(r==k) for k in range(4)] for r in range(4)],
                 bias=[0]*4,multiplier=1,shift=0,activation='none',input_scale=1.0,output_scale=1.0)]
    with tempfile.TemporaryDirectory(prefix='tiny-npu-tb-') as temp:
        root=Path(temp)
        c=compile_model(layers,4)
        write_hex(root/'sram.hex',c.image(encode_program(c.instructions)))
        command=[str(args.binary.resolve()),'--sram',str(root/'sram.hex'),'--batch','4']
        def run(extra=(),text=None,fail=False):
            result=subprocess.run(command+list(extra),input=text,capture_output=True,text=True)
            assert bool(result.returncode)==fail,result.stderr
            return result.stdout
        log=run(['--runs','100','--seed','42'])
        assert log==run(), 'Default runs/seed must remain 100/42'
        assert log!=run(['--runs','100','--seed','43'])
        assert log.count('Inference ')==100
        chunks=log.split('Input (quantized INT8):\n')[1:]
        assert len(chunks)==100
        for chunk in chunks:
            rows=[list(map(int,row.split(', '))) for row in re.findall(r'vector \d+: \[([^\]]+)\]',chunk)[:4]]
            output=re.search(r'OUTPUT beat=0 hex=([0-9A-F]{32})',chunk).group(1)
            expected=oracle(layers,np.asarray(rows))[0][-1]
            assert int(output,16).to_bytes(16,'little')==pack_tensor(np.asarray(expected))
        metrics=re.findall(r'METRICS.*',log)
        assert len(set(metrics))==1, 'Reset must restore the same timing/counters'
        interactive=run(['--interactive'],('1 2 3 4\n'*4)*2+'q\n')
        assert interactive.count('Inference ')==2
        assert len(set(re.findall(r'OUTPUT beat=0 hex=([0-9A-F]{32})',interactive)))==1
        run(['--interactive'],'128 0 0 0\n',fail=True)
        run(['--seed','4294967296'],fail=True)
        print('TB: 100 random runs, seed/default reproducibility, every output, reset, interactive and invalid INT8 checks passed')


if __name__=='__main__': main()
