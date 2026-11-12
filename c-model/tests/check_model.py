#!/usr/bin/env python3
"""Self-contained ISA/cycle/numerical checks; no third-party dependencies.
Optional --compiler ../compiler also checks current compiler-emitted images.
"""
import argparse
from fractions import Fraction
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import tempfile


def memory_constants():
    text = (Path(__file__).resolve().parents[1] / 'datapath.h').read_text()
    return {name: int(value, 16) for name, value in
            re.findall(r'^#define (\w+) (0x[0-9A-Fa-f]+)u$', text, re.M)}


MAP = memory_constants()
SRAM_BYTES = MAP['SRAM_BYTES']
SRAM_LINES = SRAM_BYTES // 16
ENTRY_PC = MAP['ENTRY_PC']
X, B, CFG = MAP['ACTIVATION_BASE'], MAP['BIAS_BASE'], MAP['CONFIG_BASE']


def instruction(op, src=0, dest=0, dims=0, flags=0):
    return ((op << 124) | (src << 92) | (dest << 60) | (dims << 32) | flags).to_bytes(16, 'little')


def pack(rows):
    return bytes((rows[v][k] if v < len(rows) and k < len(rows[v]) else 0) & 255
                 for v in range(4) for k in range(4))


def image(program):
    memory = bytearray(SRAM_BYTES)
    memory[ENTRY_PC:ENTRY_PC+16*len(program)] = b''.join(program)
    return memory


def write_hex(path, memory):
    path.write_text(''.join(f'{int.from_bytes(memory[i:i+16], "little"):032X}\n'
                            for i in range(0, len(memory), 16)))


def metric(log, name):
    return int(re.search(r'\b'+name+r'=(\d+)', log).group(1))


def rounded(value, m, shift, relu):
    return max(-128, min(127, round(Fraction(max(0, value) if relu else value, 1) * m / (1 << shift))))


class Checks:
    def __init__(self, binary, post, root):
        self.binary, self.post, self.root = binary, post, root
        self.count = 0

    def run(self, mem, data=b'', batch=4, extra=(), failure=None):
        write_hex(self.root/'sram.hex', mem)
        (self.root/'input.bin').write_bytes(data)
        result = subprocess.run([str(self.binary), '--sram', str(self.root/'sram.hex'),
            '--input-bin', str(self.root/'input.bin'), '--output-bin', str(self.root/'output.bin'),
            '--batch', str(batch), '--max-cycles', '100000', *extra], capture_output=True, text=True)
        self.count += 1
        if failure:
            assert result.returncode != 0, 'expected failure'
            assert failure in result.stderr, result.stderr
            return result
        assert result.returncode == 0, result.stderr
        return (self.root/'output.bin').read_bytes(), result.stdout

    def tile(self, batch, neurons, k, retain=False, forward=False, relu=False):
        rng = random.Random(42+batch+neurons+k)
        x = [[rng.randrange(-128,128) for _ in range(k)] for _ in range(batch)]
        w = [[rng.randrange(-16,17) for _ in range(k)] for _ in range(neurons)]
        b = [rng.randrange(-80,81) for _ in range(neurons)]
        dims = neurons | batch << 4 | k << 8
        program = [instruction(1,dest=X,dims=16)]
        for addr, field in ((B,2),(CFG,3),(0,1),(X,0)):
            program.append(instruction(8,addr,field,16))
        program.append(instruction(2,dims=dims))
        if retain:
            # Change W while retaining the current accumulator file.
            program += [instruction(8,0x10,1,16),instruction(2,dims=dims,flags=2)]
        program.append(instruction(3,dest=(X+16),dims=neurons | batch << 4,flags=4))
        if forward:
            program += [instruction(4,dest=1,dims=neurons | batch << 4),
                instruction(8,(B+32),6,16),instruction(8,(CFG+16),7,16),instruction(8,0x20,5,16),
                instruction(2,dims=neurons | batch << 4 | neurons << 8,flags=1),
                instruction(3,dest=(X+16),dims=neurons | batch << 4,flags=5)]
        program += [instruction(5,src=(X+16),dims=16),instruction(15)]
        mem=image(program);mem[:16]=pack(w)
        w2=[[rng.randrange(-16,17) for _ in range(k)] for _ in range(neurons)]
        mem[16:32]=pack(w2)
        mem[32:48]=pack([[int(r==col) for col in range(neurons)] for r in range(neurons)])
        mem[B:(B+16)]=struct.pack('<4i',*(b+[0]*(4-neurons)))
        mem[CFG:(CFG+16)]=struct.pack('<4I',3,7,int(relu),0)
        mem[(CFG+16):(CFG+32)]=struct.pack('<4I',1,0,0,0)
        expected=[[rounded(b[r]+sum(x[v][j]*(w[r][j]+(w2[r][j] if retain else 0)) for j in range(k)),3,7,relu)
                   for r in range(neurons)] for v in range(batch)]
        actual,log=self.run(mem,pack(x),batch)
        assert actual==pack(expected),(actual,expected)
        compute=k*(2 if retain else 1)+(neurons if forward else 0)
        post_count=2 if forward else 1
        loads=4+int(retain)+(3 if forward else 0)
        # Each instruction: fetch request + response + decode + update = 4;
        # Dispatch costs: load=2, MAC=K, post=1+latency, stream in=1,
        # stream out=2, forward/halt=1.
        cycles=4*len(program)+loads*2+compute+post_count*(1+self.post)+1+2+int(forward)+1
        assert metric(log,'cycles')==cycles,(metric(log,'cycles'),cycles)
        assert metric(log,'useful_macs')==compute*neurons*batch
        assert metric(log,'compute_cycles')==compute
        assert metric(log,'sram_reads')==len(program)+loads+1
        assert metric(log,'sram_writes')==1+post_count
        assert metric(log,'bus_stall_cycles')==0
        return mem,pack(x),actual,cycles

    def all(self):
        for batch in range(1,5):
            for n,k in ((4,4),(2,3),(1,1)):
                self.tile(batch,n,k,retain=True,relu=n!=2)
            self.tile(batch,3,4,forward=True)
        mem,data,expected,cycles=self.tile(4,4,4,retain=True)
        actual,log=self.run(mem,data,extra=('--input-delay','60','--output-delay','300'))
        assert actual==expected
        assert metric(log,'bus_stall_cycles')==metric(log,'cycles')-cycles
        assert metric(log,'sram_reads')==metric(self.run(mem,data)[1],'sram_reads')
        # Stream loop has two iterations and no implicit rewind or lost beats.
        p=[instruction(6,src=2,dest=(ENTRY_PC+16)),instruction(1,dest=X,dims=16),
           instruction(5,src=X,dims=16),instruction(7,dest=(ENTRY_PC+16)),instruction(15)]
        actual,log=self.run(image(p),bytes(range(32)))
        assert actual==bytes(range(32))
        assert metric(log,'instructions')==8 and metric(log,'cycles')==42
        self.run(image([instruction(9)]),failure='invalid instruction')
        self.run(image([instruction(8,src=1,dest=1,dims=16)]),failure='invalid instruction')
        self.run(image([instruction(8,src=B,dest=1,dims=16)]),failure='invalid instruction')
        self.run(image([instruction(2,dims=0x444,flags=2)]),failure='invalid operands')
        self.run(image([instruction(3,dest=X,dims=0x44,flags=4)]),failure='POST requires')
        self.run(image([instruction(4,dest=0,dims=0x44)]),failure='invalid operands')
        self.run(image([instruction(7,dest=ENTRY_PC)]),failure='invalid instruction')
        self.run(image([instruction(6,src=2,dest=(ENTRY_PC+16)),instruction(15)]),failure='invalid instruction')
        self.run(image([instruction(1,dest=X,dims=16)]),failure='cycle limit')
        self.run(image([instruction(15)]),bytes(16),failure='unused input')
        bad=bytearray(mem);bad[CFG:(CFG+4)]=bytes(4)
        self.run(bad,data,failure='invalid operands, config')
        for config in ((0x80000000,7,1,0),(3,63,1,0),(3,7,2,0),(3,7,1,1)):
            bad=bytearray(mem);bad[CFG:CFG+16]=struct.pack('<4I',*config)
            self.run(bad,data,failure='invalid operands, config')
        bad=bytearray(mem);bad[B:(B+16)]=struct.pack('<4i',*([2**31-1]*4));bad[:16]=bytes([127]*16)
        self.run(bad,bytes([127]*16),failure='INT32 overflow')
        self.run(mem,data,batch=1,failure='invalid instruction')
        # Validate preloading failures, including trailing and short words.
        for text in ('', '0\n'*SRAM_LINES, 'G'*32+'\n', '0'*32+'\n'*0, ('0'*32+'\n')*(SRAM_LINES+1)):
            (self.root/'bad.hex').write_text(text)
            p=subprocess.run([str(self.binary),'--sram',str(self.root/'bad.hex'),'--input-bin',str(self.root/'input.bin')],capture_output=True,text=True)
            assert p.returncode and 'invalid SRAM hex' in p.stderr
            self.count+=1

    def loader(self, checker):
        # Every byte, including 0xFFFF0, must load in low-address/rightmost order.
        memory = bytes(range(256)) * (SRAM_BYTES // 256)
        (self.root/'all.bin').write_bytes(memory)
        write_hex(self.root/'all.hex', memory)
        text = (self.root/'all.hex').read_text()
        lines = text.splitlines()
        assert len(lines)==SRAM_LINES and all(len(line)==32 for line in lines)
        def check(path, expected):
            p=subprocess.run([str(checker),str(path),str(expected)],capture_output=True,text=True)
            assert p.returncode==0,p.stderr
            self.count+=1
        check(self.root/'all.hex', self.root/'all.bin')
        commented='# header\n\n// header\n'+'\n'.join('  '+line+(' # word' if i%2 else ' // word') for i,line in enumerate(lines))
        (self.root/'comments.hex').write_text(commented)
        check(self.root/'comments.hex', self.root/'all.bin')
        for bad in (text[:-33], text+'0'*32+'\n', text[:-2]+'G\n',
                    text+'trailing garbage\n', ' '*240+'0\n', text[:-33]+'0'*31+'\n'):
            (self.root/'bad.hex').write_text(bad)
            check(self.root/'bad.hex', 'reject')
        check(self.root/'missing.hex','reject')
        check(self.root,'reject')  # directory read failure, not just fopen failure
        print('Loader: byte-for-byte full image, comments, final beat and transactional rejection passed')

    def compiler(self, directory, baseline=None):
        sys.path.insert(0,str(directory.resolve()))
        from compiler import compile_model, write_hex
        from isa import encode_program, REGIONS, Instruction, Op
        from verify import execute, pack_tensor, oracle, unpack_tensor
        import numpy as np
        assert REGIONS=={'weights':(MAP['WEIGHT_BASE'],MAP['WEIGHT_END']),
                        'activations':(MAP['ACTIVATION_BASE'],MAP['ACTIVATION_END']),
                        'bias':(MAP['BIAS_BASE'],MAP['BIAS_END']),
                        'config':(MAP['CONFIG_BASE'],MAP['CONFIG_END']),
                        'program':(ENTRY_PC,SRAM_BYTES)}
        rng=random.Random(42)
        for shape in ((4,4,4),(4,8,8,2),(5,6,3),(3,2),(4,3,8),(4,100,100,1)):
            layers=[]
            for li,(k,n) in enumerate(zip(shape,shape[1:])):
                layers.append(dict(weights=[[rng.randrange(-12,13) for _ in range(k)] for _ in range(n)],
                    bias=[rng.randrange(-100,101) for _ in range(n)], multiplier=3,shift=7,
                    activation='relu' if li<len(shape)-2 else 'none',input_scale=1.0,output_scale=1.0))
            for batch in range(1,5):
                x=np.array([[rng.randrange(-128,128) for _ in range(shape[0])] for _ in range(batch)])
                for forwarding in (False,True):
                    compilation=compile_model(layers,batch,forwarding)
                    memory=compilation.image(encode_program(compilation.instructions))
                    data=pack_tensor(x);reference=execute(memory,data)
                    actual,log=self.run(memory,data,batch)
                    assert actual==reference['output']
                    assert unpack_tensor(actual,batch,shape[-1]).tolist()==oracle(layers,x)[0][-1]
                    independent_layers,partials=oracle(layers,x)
                    independent={(p['layer'],p['k']):p['accumulators'] for p in partials}
                    schedule={s['pc']:s for s in compilation.schedule}
                    for event in reference['trace']:
                        meta=schedule[event['pc']]
                        for step in event.get('steps',[]):
                            offset=meta['neuron_offset'];rows=len(step['accumulators'])
                            assert step['accumulators']==independent[meta['layer'],meta['k_offset']+step['local_k']][offset:offset+rows]
                        if 'output' in event:
                            offset=meta['neuron_offset'];width=len(event['output'][0])
                            assert event['output']==[row[offset:offset+width] for row in independent_layers[meta['layer']]]
                    assert metric(log,'instructions')==reference['instructions']
                    assert metric(log,'useful_macs')==reference['useful_macs']
                    steps=[(event['pc'],s) for event in reference['trace'] for s in event.get('steps',[])]
                    pattern=r'PE\s+pc=0x([0-9A-F]+) k=(\d+) r=(\d+) v=(\d+) acc=(-?\d+)'
                    observed=re.findall(pattern,log)
                    if not observed:
                        raise AssertionError('--compiler comparison requires LOG=flow or LOG=both')
                    expected=[]
                    for pc,step in steps:
                        for r,row in enumerate(step['accumulators']):
                            for v,value in enumerate(row): expected.append((pc,step['local_k'],r,v,value))
                    assert [(int(p,16),int(k),int(r),int(v),int(a)) for p,k,r,v,a in observed]==expected
                    dispatch=sum((2 if i.op in (5,8) else (i.dims>>8)&15 if i.op==2 else 1+self.post if i.op==3 else 1)
                                 for i in compilation.instructions)
                    assert metric(log,'cycles')==4*len(compilation.instructions)+dispatch
                    if baseline and shape!=(4,100,100,1):
                        # Run exactly the same program in the pre-change model,
                        # translating addresses only. Compare outputs and timing.
                        old_regions={'weights':(0,0x2000),'activations':(0x2000,0x3000),
                                     'bias':(0x3000,0x3800),'config':(0x3800,0x3900)}
                        old=bytearray(0x4000)
                        for region,(start,end) in old_regions.items():
                            new=REGIONS[region][0];old[start:end]=memory[new:new+end-start]
                        for index,ins in enumerate(compilation.instructions):
                            src,dest=ins.src,ins.dest
                            if ins.op==Op.LOAD_OPERAND:
                                region=('activations','weights','bias','config')[dest&3]
                                src+=old_regions[region][0]-REGIONS[region][0]
                            if ins.op==Op.STREAM_STORE: src+=0x2000-REGIONS['activations'][0]
                            if ins.op==Op.STREAM_LOAD or (ins.op==Op.POST_PROC and ins.flags&4):
                                dest+=0x2000-REGIONS['activations'][0]
                            old[0x3900+16*index:0x3910+16*index]=instruction(int(ins.op),src,dest,ins.dims,ins.flags)
                        current=self.binary
                        try:
                            self.binary=baseline.resolve();old_output,old_log=self.run(old,data,batch)
                        finally: self.binary=current
                        assert old_output==actual
                        for name in ('cycles','instructions','useful_macs','compute_cycles','bus_stall_cycles','sram_reads','sram_writes'):
                            assert metric(log,name)==metric(old_log,name),(shape,batch,name)
                    if shape==(4,8,8,2) and batch==4 and not forwarding:
                        print('Compiler 4->8->8->2:',re.search(r'METRICS.*',log).group())
        print('Compiler integration: final output, independent tensor oracle, every PE accumulator and cycle formulas passed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--post-latency',type=int,default=1,choices=(0,1));p.add_argument('--compiler',type=Path)
    p.add_argument('--loader-checker',type=Path);p.add_argument('--baseline-binary',type=Path)
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='tiny-npu-check-') as temporary:
        checks=Checks(args.binary.resolve(),args.post_latency,Path(temporary));checks.all()
        if args.loader_checker: checks.loader(args.loader_checker.resolve())
        if args.compiler: checks.compiler(args.compiler,args.baseline_binary)
        print(f'{checks.count} model checks passed (POST_LATENCY={args.post_latency})')

if __name__=='__main__': main()
