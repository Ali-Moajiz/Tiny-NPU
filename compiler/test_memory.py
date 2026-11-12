"""Memory-map and image-format regressions for the 1 MiB SRAM ABI."""
from pathlib import Path
import tempfile
import unittest

from compiler import Image, compile_model, write_outputs
from isa import ENTRY_PC, SRAM_BYTES, REGIONS, Instruction, Op, address, encode_program


class MemoryTests(unittest.TestCase):
    def test_map_and_boundaries(self):
        self.assertEqual(SRAM_BYTES, 1048576)
        self.assertEqual(REGIONS, {'weights':(0x00000,0xC0000),
                                  'activations':(0xC0000,0xE0000),
                                  'bias':(0xE0000,0xF0000),
                                  'config':(0xF0000,0xF1000),
                                  'program':(0xF1000,0x100000)})
        last = 0
        for region, (start, end) in REGIONS.items():
            self.assertEqual(start, last)
            self.assertEqual(start % 16, 0)
            self.assertEqual(end % 16, 0)
            self.assertGreater(end, start)
            address(start, region)
            address(end - 16, region)
            for bad in (start - 16, start + 1, end - 15, end, SRAM_BYTES):
                with self.subTest(region=region, bad=bad), self.assertRaises(ValueError):
                    address(bad, region)
            last = end
        self.assertEqual(last, SRAM_BYTES)
        self.assertEqual(sum(end-start for start,end in REGIONS.values()), SRAM_BYTES)
        self.assertEqual(ENTRY_PC, REGIONS['program'][0])
        Instruction(Op.LOOP_END, dest=SRAM_BYTES-16).validate()

    def test_every_region_capacity(self):
        for region, (start, end) in REGIONS.items():
            image = Image()
            self.assertEqual(image.allocate(region, 'full', bytes(end-start)), start)
            with self.assertRaisesRegex(ValueError, f'{region} capacity'):
                image.allocate(region, 'overflow', bytes(16))
            for size in (0, 1, 15, 17):
                with self.assertRaises(ValueError):
                    Image().allocate(region, 'unaligned', bytes(size))

    def test_large_weights_and_full_image(self):
        # 4->100->100->1 exceeds the former weight and program capacities.
        widths = [4, 100, 100, 1]
        layers = [dict(name=f'L{i}', weights=[[1]*k for _ in range(n)],
                       bias=[0]*n, multiplier=1, shift=8, activation='none',
                       input_scale=1.0, output_scale=1.0)
                  for i,(k,n) in enumerate(zip(widths,widths[1:]))]
        c = compile_model(layers)
        weights = [a for a in c.manifest['allocations'] if a['region']=='weights']
        self.assertEqual(sum(a['size'] for a in weights), 10800)
        self.assertGreater(c.manifest['instruction_count'], 112)
        self.assertLessEqual(c.manifest['instruction_count'], 3840)
        self.assertTrue(all(REGIONS['weights'][0] <= a['address'] and
                            a['address']+a['size'] <= REGIONS['weights'][1] for a in weights))
        program = encode_program(c.instructions)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_outputs(c, program, root, {'mode':'test', 'source_input_dtype':'float32',
                          'source_output_dtype':'float32'}, 'test.onnx', debug=True)
            binary = (root/'sram.bin').read_bytes()
            lines = (root/'sram.hex').read_text().splitlines()
            self.assertEqual(len(binary), SRAM_BYTES)
            self.assertEqual(len(lines), 65536)
            self.assertTrue(all(len(line)==32 and all(ch in '0123456789ABCDEF' for ch in line) for line in lines))
            self.assertEqual(b''.join(int(line,16).to_bytes(16,'little') for line in lines), binary)
            self.assertEqual((root/'params.bin').stat().st_size, ENTRY_PC)
            self.assertEqual((root/'program.bin').read_bytes(), program)
            self.assertEqual(len((root/'program.hex').read_text().splitlines()), len(c.instructions))
            self.assertEqual(binary[ENTRY_PC:ENTRY_PC+len(program)], program)


if __name__ == '__main__':
    unittest.main()
