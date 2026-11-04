#!/usr/bin/env python3
"""Independent Python integer oracle plus edge-by-edge architecture regressions.
Requires Python standard library and a C99 compiler; no torch/numpy/HLS tools.
Temporary shared library stays outside the source tree.
"""
import ctypes as C
from fractions import Fraction
import json
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import tempfile

LATENCY=int(os.environ.get("POST_LATENCY", "1"))
ROOT=Path(__file__).resolve().parents[1]
SOURCES="logging.c controller.c sram.c reg128.c mac.c scale_clamp.c bus.c framework.c npu_top.c".split()
BRIDGE=r"""
#include "npu.h"
#include <stdlib.h>
void *create(unsigned batch,unsigned prefetch,unsigned fuse) {
    Npu *n=calloc(1,sizeof *n);
    if(!n || !npu_init_options(n,(NpuOptions){batch,prefetch!=0,fuse!=0})) {free(n);return NULL;}
    return n;
}
void destroy(void *p) {free(p);}
void put(void *p,unsigned addr,unsigned lane,uint32_t word) {((Npu *)p)->data.sram.regs.lines[addr].lane[lane]=word;}
uint32_t get(void *p,unsigned addr,unsigned lane) {return ((Npu *)p)->data.sram.regs.lines[addr].lane[lane];}
int step(void *p) {return npu_run(p,1);}
uint64_t metric(void *p,unsigned key) {
    Npu *n=p;Control c=n->buses.control.payload;
    switch(key) {
    case 0:return n->cycle-1;case 1:return n->data.sram.regs.reads;case 2:return n->data.sram.regs.writes;
    case 3:return n->data.array.regs.updates;case 4:return n->data.array.regs.compute_cycles;case 5:return n->data.array.regs.peak_active;
    case 6:return n->controller.regs.overlap_reads;case 7:return n->data.reg_i.regs.forwards;
    case 8:return c.compute_en?2:0;case 9:return c.k+1;case 10:return c.bank;case 11:return c.read_enable;
    case 12:return c.read_bank;case 13:return c.rd_addr;case 14:return c.layer;
    default:return 0;
    }
}
uint32_t input_bank(void *p,unsigned bank,unsigned lane) {return ((Npu *)p)->data.reg_i.regs.banks.bank[bank].line.lane[lane];}
uint32_t weight_bank(void *p,unsigned bank,unsigned lane) {return ((Npu *)p)->data.reg_w.regs.banks.bank[bank].line.lane[lane];}
int32_t acc(void *p,unsigned r,unsigned v) {return ((Npu *)p)->data.array.regs.sums.value[r][v];}
int8_t requant(int32_t x,int32_t m,unsigned s) {return scale_clamp_value(x,m,s);}
void bad_address(Npu *n) {n->controller.words[0].rd_addr=1024;}
void active_bank_collision(Npu *n) {n->controller.regs.tag.bank=0;}
int rejected_schedule(Npu *n, unsigned fault) {
    Controller c=n->controller;
    if(fault==0) c.words[0].store_en=true;
    if(fault==1) c.words[0].rd_addr=1024;
    if(fault==2) c.words[4].read_bank=0,c.words[4].rd_addr=0;
    return !controller_validate(&c);
}
unsigned signature(Npu *n) {
    Control c=n->buses.control.payload;
    unsigned we=c.we_weight+2*c.we_input+4*c.we_bias+8*c.we_cfg;
    return (c.compute_en?1+c.k+4*c.layer:0) |
           (c.read_enable?(1+c.read_type+4*c.read_bank)<<4:0) |
           (we?(we+16*c.capture_bank)<<8:0) |
           ((unsigned)c.post_en<<13) | ((unsigned)c.forward_en<<14) |
           ((unsigned)c.store_en<<15);
}

"""

def arrays():
    text=(ROOT/'trained_npu_params.h').read_text()
    result={}
    for name in ['trained_weights','trained_biases','requant_multiplier','requant_shift','relu_enable']:
        literal=re.search(r'\b'+name+r'\s*(?:\[[^]]*\])+\s*=\s*(.*?);',text,re.S).group(1)
        result[name]=json.loads(literal.replace('{','[').replace('}',']'))
    return result

P=arrays()
def quant(x,m,s):return max(-128,min(127,round(Fraction(x*m,1<<s))))
def reference(vector):
    stages=[];x=list(vector)
    for l in range(2):
        partial=[];sums=list(P['trained_biases'][l])
        for k in range(4):
            sums=[sums[r]+P['trained_weights'][l][r][k]*x[k] for r in range(4)]
            assert all(-(1<<31)<=v<(1<<31) for v in sums)
            partial.append(list(sums))
        act=[max(0,v) if P['relu_enable'][l] else v for v in sums]
        x=[quant(v,P['requant_multiplier'][l],P['requant_shift'][l]) for v in act]
        stages.append((partial,list(x)))
    return stages

def pack(x):return sum((v&255)<<(8*i) for i,v in enumerate(x))
def unpack(x):return [((x>>(8*i))&255)-256 if ((x>>(8*i))&255)>=128 else ((x>>(8*i))&255) for i in range(4)]


def main():
    rng=random.Random(42);case_count=vector_count=0;times={}
    with tempfile.TemporaryDirectory(prefix='tiny-npu-check-') as td:
        td=Path(td);(td/'bridge.c').write_text(BRIDGE)
        cmd=shlex.split(os.environ.get('CC','cc'))+['-std=c99','-O1','-g','-Wall','-Wextra','-Werror','-fPIC','-shared',
             '-fsanitize=undefined','-fno-sanitize-recover=all',f'-DPOST_LATENCY={LATENCY}','-DNPU_LOG_FLOW=0','-DNPU_LOG_PORTS=0','-I',str(ROOT),str(td/'bridge.c')]
        cmd += [str(ROOT/s) for s in SOURCES]+['-o',str(td/'check.so')]
        subprocess.run(cmd,check=True)
        lib=C.CDLL(str(td/'check.so'))
        lib.create.argtypes=[C.c_uint]*3;lib.create.restype=C.c_void_p
        lib.destroy.argtypes=[C.c_void_p]
        for name in ['put'] :getattr(lib,name).argtypes=[C.c_void_p,C.c_uint,C.c_uint,C.c_uint32]
        for name in ['get','input_bank','weight_bank']:
            fn=getattr(lib,name);fn.argtypes=[C.c_void_p,C.c_uint,C.c_uint];fn.restype=C.c_uint32
        lib.step.argtypes=[C.c_void_p];lib.step.restype=C.c_int
        lib.metric.argtypes=[C.c_void_p,C.c_uint];lib.metric.restype=C.c_uint64
        lib.acc.argtypes=[C.c_void_p,C.c_uint,C.c_uint];lib.acc.restype=C.c_int32
        lib.requant.argtypes=[C.c_int32,C.c_int32,C.c_uint];lib.requant.restype=C.c_int8
        for name in ['bad_address','active_bank_collision']:getattr(lib,name).argtypes=[C.c_void_p]

        def check(vectors,prefetch,fuse):
            nonlocal case_count,vector_count
            batch=len(vectors);n=lib.create(batch,prefetch,fuse);assert n
            try:
                refs=[reference(x) for x in vectors]
                for v,x in enumerate(vectors):lib.put(n,0x200,v,pack(x))
                for v in range(batch,4):lib.put(n,0x200,v,0xA5A50000+v)
                for v in range(4):lib.put(n,0x201,v,0x5A5A0000+v)
                shadow_weight=[pack(row) for row in P['trained_weights'][0]]
                trace=[]
                for _ in range(128):
                    before=lib.metric(n,3);status=lib.step(n);assert status!=0
                    kind,k,bank,read,read_bank,address,layer=[lib.metric(n,key) for key in range(8,15)]
                    trace.append((kind,k,layer))
                    if kind==2: # EDGE_COMPUTE
                        assert lib.metric(n,3)-before==4*batch
                        assert not read or read_bank!=bank
                        for r in range(4):
                            for v in range(batch):assert lib.acc(n,r,v)==refs[v][layer][0][k-1][r]
                            for v in range(batch,4):assert lib.acc(n,r,v)==0
                        if layer==0:
                            for r in range(4):assert lib.weight_bank(n,0,r)==shadow_weight[r]
                    else:assert lib.metric(n,3)==before
                    if fuse and layer==1 and kind==2:
                        for v in range(batch):assert lib.input_bank(n,1,v)==pack(refs[v][0][1])
                    if status==1:break
                else:raise AssertionError('controller did not finish')
                assert lib.metric(n,3)==32*batch and lib.metric(n,4)==8 and lib.metric(n,5)==4*batch
                assert lib.metric(n,1)==(7 if fuse else 8)
                assert lib.metric(n,2)==(1 if fuse else 2)
                assert lib.metric(n,6)==(3 if prefetch else 0)
                assert lib.metric(n,7)==(1 if fuse else 0)
                for v in range(batch):assert unpack(lib.get(n,0x200,v))==refs[v][1][1]
                for v in range(batch,4):assert lib.get(n,0x200,v)==0xA5A50000+v
                for v in range(4):
                    expected=pack(refs[v][0][1]) if not fuse and v<batch else 0x5A5A0000+v
                    assert lib.get(n,0x201,v)==expected
                # 4 initial reads, 8 MACs, 2 posts, and one drain per
                # post register. Unfused reload costs two edges with
                # prefetch. Otherwise it shares capture with parameter reads.
                extra = 2*(not fuse) if prefetch else 3+(not fuse)
                cycles=lib.metric(n,0)
                assert cycles==4+8+2+2*LATENCY+extra
                times.setdefault((prefetch,fuse),cycles);assert times[prefetch,fuse]==cycles
                case_count+=1;vector_count+=batch
            finally:lib.destroy(n)

        fixtures=json.loads((ROOT/'npu_reference_vectors.json').read_text())
        # Existing exported examples (the JSON also preserves the original source metadata).
        examples=fixtures['vectors'] if isinstance(fixtures,dict) and 'vectors' in fixtures else fixtures
        # Deterministic boundary/mapping vectors plus randomized, distinct batches.
        boundaries=[item['input_q'] for item in examples]
        for prefetch in (0,1):
            for fuse in (0,1):
                for x in boundaries:check([x],prefetch,fuse)
                for batch in range(1,5):
                    for _ in range(64):check([[rng.randrange(-128,128) for _ in range(4)] for _ in range(batch)],prefetch,fuse)
        # Compare notebook fixtures to the independent oracle too, if the export uses this layout.
        if isinstance(examples,list):
            for item in examples:
                assert reference(item['input_q'])[-1][1]==item['expected_output_q']
        # Requantization boundary/tie tests, including every legal shift.
        for shift in range(63):
            for m in [1,3,1585768005,2147483647]:
                for x in [-2147483648,-257,-129,-3,-1,0,1,3,127,255,2147483647]:
                    assert lib.requant(x,m,shift)==quant(x,m,shift)
        assert not lib.create(0,1,1) and not lib.create(5,1,1)
        def fault(setup,steps=128,before=None):
            n=lib.create(1,1,1);assert n
            try:
                setup(n)
                for i in range(steps):
                    if before:before(n,i+1)
                    if lib.step(n)==0:
                        assert i+1==steps and lib.metric(n,0)==i
                        assert lib.step(n)==0 and lib.metric(n,0)==i
                        return
                raise AssertionError('fault not rejected')
            finally:lib.destroy(n)
        fault(lib.bad_address,1)
        for lane,value in [(0,0),(0,0x80000000),(1,63),(2,2),(3,1)]:
            fault(lambda n,lane=lane,value=value:lib.put(n,0x380,lane,value),5)
        fault(lambda n:lib.put(n,0x300,0,0x7fffffff),5)
        fault(lambda n:None,6,lambda n,cycle:lib.active_bank_collision(n) if cycle==6 else None)
        lib.rejected_schedule.argtypes=[C.c_void_p,C.c_uint]
        lib.signature.argtypes=[C.c_void_p]
        lib.signature.restype=C.c_uint
        n=lib.create(4,1,1)
        assert n
        try:
            for f in range(3):
                assert lib.rejected_schedule(n,f)
            # Manual default trace. The fields describe compute wave,
            # encoded read, capture mask with bank, post, forward, store.
            head=[
                (0,1,0,0,0,0), (0,2,1,0,0,0), (0,3,2,0,0,0),
                (0,4,4,0,0,0), (1,5,8,0,0,0), (2,7,17,0,0,0),
                (3,8,20,0,0,0), (4,0,24,0,0,0),
            ]
            if LATENCY:
                head += [(0,0,0,1,0,0),(0,0,0,0,1,0)]
            else:
                head += [(0,0,0,1,1,0)]
            head += [(k,0,0,0,0,0) for k in range(5,9)]
            if LATENCY:
                head += [(0,0,0,1,0,0),(0,0,0,0,0,1)]
            else:
                head += [(0,0,0,1,0,1)]
            for i,(mac,rd,we,post,fwd,wr) in enumerate(head):
                assert lib.step(n)==(1 if i+1==len(head) else 2)
                assert lib.signature(n)==mac+(rd<<4)+(we<<8)+(post<<13)+(fwd<<14)+(wr<<15),i+1
        finally:
            lib.destroy(n)
        print(f'PASS: POST_LATENCY={LATENCY}, {len(head)} hand-checked batch-4 edges and schedule fault injection.')
        # Deliberate faults above emit expected diagnostics to stderr.
    print(f'PASS: {case_count} batch inferences / {vector_count} independent vectors; all intermediate MACs and final outputs match.')
    print('PASS: masked writes, old-value register timing, bank isolation, hidden bypass, single-port overlap and fault checks.')
    print('PASS: integer quantization boundary/tie tests for all shifts 0..62 (UBSan enabled).')
    for (prefetch,fuse),cycles in sorted(times.items()):print(f'  PREFETCH={prefetch} FUSE={fuse}: {cycles} cycles per batch of 1..4 vectors')
if __name__=='__main__':main()
