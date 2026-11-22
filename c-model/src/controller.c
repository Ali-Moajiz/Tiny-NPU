#include "controller.h"
#include "sim.h"
#include <stdio.h>
#include <string.h>

/* ---- Instruction decode ---- */

Instruction instruction_decode(Line128 r)
{
    Instruction i;
    i.opcode = r.lane[3] >> 28;
    i.src = (r.lane[2] >> 28) | ((r.lane[3] & 0x0fffffffU) << 4);
    i.dest = (r.lane[1] >> 28) | ((r.lane[2] & 0x0fffffffU) << 4);
    i.dims = r.lane[1] & 0x0fffffffU;
    i.flags = r.lane[0];
    return i;
}

/* ---- Instruction handlers and registry ---- */

/* ISA-specific callbacks. Add a descriptor to builtins[] to extend the ISA;
 * controller.c, SRAM, and the cycle framework do not need an opcode switch. */
static bool dimensions(Instruction i, bool mac)
{
    unsigned n = instruction_n(i), v = instruction_v(i), k = instruction_k(i);
    return !(i.dims >> 12) && n >= 1 && n <= 4 && v >= 1 && v <= 4 && (mac ? (k >= 1 && k <= 4) : k == 0);
}
static bool validate_stream_load(Instruction i)
{
    return !i.src && !i.flags && i.dims == 16 && isa_address(i.dest, ACTIVATION_BASE, ACTIVATION_END);
}
static bool validate_stream_store(Instruction i)
{
    return !i.dest && !i.flags && i.dims == 16 && isa_address(i.src, ACTIVATION_BASE, ACTIVATION_END);
}
static bool validate_load_operand(Instruction i)
{
    static const uint32_t begin[] = {ACTIVATION_BASE, WEIGHT_BASE, BIAS_BASE, CONFIG_BASE},
                          end[] = {ACTIVATION_END, WEIGHT_END, BIAS_END, CONFIG_END};
    return i.dest <= 7 && !i.flags && i.dims == 16 && isa_address(i.src, begin[i.dest & 3], end[i.dest & 3]);
}
static bool validate_mac(Instruction i)
{
    return dimensions(i, true) && !i.src && !i.dest && !(i.flags & ~3u);
}
static bool validate_post(Instruction i)
{
    return dimensions(i, false) && !i.src && !(i.flags & ~5u) && ((i.flags & 4) ? isa_address(i.dest, ACTIVATION_BASE, ACTIVATION_END) : !i.dest);
}
static bool validate_forward(Instruction i)
{
    return dimensions(i, false) && !i.src && i.dest < 2 && !i.flags;
}
static bool validate_loop_start(Instruction i)
{
    return i.src && !i.dims && !i.flags && isa_address(i.dest, ENTRY_PC, SRAM_BYTES);
}
static bool validate_loop_end(Instruction i)
{
    return !i.src && !i.dims && !i.flags && isa_address(i.dest, ENTRY_PC, SRAM_BYTES);
}
static bool validate_halt(Instruction i)
{
    return !i.src && !i.dest && !i.dims && !i.flags;
}
static void tile_control(const Controller *c, Control *out)
{
    Instruction i = c->regs.instruction;
    out->slot = i.flags & 1;
    out->neurons = instruction_n(i);
    out->batch = instruction_v(i);
}
static bool batch_matches(const Controller *c, Control out)
{
    return !c->expected_batch || c->expected_batch == out.batch;
}
static ExecStatus stream_load(const Controller *c, const ControllerInputs *in, Control *out)
{
    if (!in->stream.input_valid)
        return EXEC_WAIT;
    out->wr_en = true;
    out->wr_addr = c->regs.instruction.dest;
    out->byte_enable = 0xffff;
    out->write_source = WRITE_STREAM;
    out->input_pop = true;
    return EXEC_DONE;
}
static ExecStatus load_operand(const Controller *c, const ControllerInputs *in, Control *out)
{
    Instruction i = c->regs.instruction;
    if (!c->regs.step)
    {
        out->rd_en = true;
        out->rd_addr = i.src;
        return EXEC_PROGRESS;
    }
    if (!in->memory.valid)
        return EXEC_WAIT;
    out->read_consume = true;
    out->latch_en = true;
    out->latch_slot = i.dest >> 2;
    out->latch_field = i.dest & 3;
    return EXEC_DONE;
}
static ExecStatus exec_mac(const Controller *c, const ControllerInputs *in, Control *out)
{
    (void)in;
    tile_control(c, out);
    if (!batch_matches(c, *out))
        return EXEC_FAULT;
    out->compute_en = true;
    out->accumulate = (c->regs.instruction.flags & 2) != 0;
    out->k = c->regs.step;
    out->k_count = instruction_k(c->regs.instruction);
    return out->k + 1 == out->k_count ? EXEC_DONE : EXEC_PROGRESS;
}
static ExecStatus post_proc(const Controller *c, const ControllerInputs *in, Control *out)
{
    (void)in;
    tile_control(c, out);
    if (!batch_matches(c, *out))
        return EXEC_FAULT;
    /* Read one neuron column per clock into the four activation lanes.
     * Drain activation + scaling before the packer's last capture. A store
     * sees that registered tile on the following clock. */
    out->post_en = c->regs.step < out->neurons;
    out->post_col = c->regs.step;
    unsigned last_capture = out->neurons + POST_PIPELINE_STAGES - 1;
    bool store = (c->regs.instruction.flags & 4) != 0;
    if (c->regs.step < last_capture + (unsigned)store)
        return EXEC_PROGRESS;
    if (store)
    {
        out->wr_en = true;
        out->wr_addr = c->regs.instruction.dest;
        out->byte_enable = 0xffff;
        out->write_source = WRITE_RESULT;
    }
    return EXEC_DONE;
}
static ExecStatus forward(const Controller *c, const ControllerInputs *in, Control *out)
{
    (void)in;
    tile_control(c, out);
    if (!batch_matches(c, *out))
        return EXEC_FAULT;
    out->forward_en = true;
    out->forward_slot = c->regs.instruction.dest;
    return EXEC_DONE;
}
static ExecStatus stream_store(const Controller *c, const ControllerInputs *in, Control *out)
{
    if (!c->regs.step)
    {
        out->rd_en = true;
        out->rd_addr = c->regs.instruction.src;
        return EXEC_PROGRESS;
    }
    if (!in->memory.valid || !in->stream.output_ready)
        return EXEC_WAIT;
    out->read_consume = true;
    out->output_push = true;
    return EXEC_DONE;
}
static ExecStatus loop_start(const Controller *c, const ControllerInputs *in, Control *out)
{
    (void)in;
    (void)out;
    return c->regs.loop_active || c->regs.instruction.dest != c->regs.pc + 16 ? EXEC_FAULT : EXEC_DONE;
}
static void retire_loop_start(Controller *c)
{
    c->regs.loop_active = true;
    c->regs.loop_remaining = c->regs.instruction.src;
    c->regs.loop_pc = c->regs.instruction.dest;
}
static ExecStatus loop_end(const Controller *c, const ControllerInputs *in, Control *out)
{
    (void)in;
    (void)out;
    return !c->regs.loop_active || c->regs.loop_pc != c->regs.instruction.dest ? EXEC_FAULT : EXEC_DONE;
}
static void retire_loop_end(Controller *c)
{
    if (--c->regs.loop_remaining)
        c->regs.next_pc = c->regs.loop_pc;
    else
        c->regs.loop_active = false;
}
static ExecStatus halt(const Controller *c, const ControllerInputs *in, Control *out)
{
    (void)in;
    (void)out;
    return c->regs.loop_active ? EXEC_FAULT : EXEC_DONE;
}
static void retire_halt(Controller *c) { c->regs.done = true; }
bool controller_register_builtins(Controller *c)
{
    static const InstructionHandler builtins[] = {
        {OP_STREAM_LOAD, "STREAM_LOAD", validate_stream_load, stream_load, NULL},
        {OP_EXEC_MAC, "EXEC_MAC", validate_mac, exec_mac, NULL},
        {OP_POST_PROC, "POST_PROC", validate_post, post_proc, NULL},
        {OP_FORWARD, "FORWARD", validate_forward, forward, NULL},
        {OP_STREAM_STORE, "STREAM_STORE", validate_stream_store, stream_store, NULL},
        {OP_LOOP_START, "LOOP_START", validate_loop_start, loop_start, retire_loop_start},
        {OP_LOOP_END, "LOOP_END", validate_loop_end, loop_end, retire_loop_end},
        {OP_LOAD_OPERAND, "LOAD_OPERAND", validate_load_operand, load_operand, NULL},
        {OP_HALT, "HALT", validate_halt, halt, retire_halt}};
    for (size_t i = 0; i < sizeof builtins / sizeof builtins[0]; i++)
        if (!controller_register(c, &builtins[i]))
            return false;
    return true;
}

/* ---- Fetch / decode / dispatch / update sequencer ---- */

void controller_init(Controller *c)
{
    memset(c, 0, sizeof *c);
    c->regs.pc = ENTRY_PC;
    c->regs.state = CTRL_FETCH;
    (void)controller_register_builtins(c);
}
bool controller_register(Controller *c, const InstructionHandler *h)
{
    if (!h || h->opcode > 15 || !h->name || !h->validate || !h->execute || c->handlers[h->opcode])
        return false;
    c->handlers[h->opcode] = h;
    return true;
}
static bool evaluate(void *instance, uint64_t cycle)
{
    Controller *c = instance;
    Control out = {0};
    out.pc = c->regs.pc;
    ControllerInputs in = {TileSignal_read(c->ports.in.memory), StreamStatus_read(c->ports.in.stream)};
    c->execution = EXEC_PROGRESS;
    switch (c->regs.state)
    {
    case CTRL_FETCH:
        if (!isa_address(c->regs.pc, ENTRY_PC, SRAM_BYTES))
            goto fault;
        out.rd_en = true;
        out.rd_addr = c->regs.pc;
        break;
    case CTRL_RESPONSE:
        if (in.memory.valid)
            out.read_consume = true;
        else
            c->execution = EXEC_WAIT;
        break;
    case CTRL_DECODE:
    {
        c->decoded = instruction_decode(c->regs.ir);
        const InstructionHandler *h = c->handlers[c->decoded.opcode];
        if (!h || !h->validate(c->decoded))
            goto fault;
        break;
    }
    case CTRL_DISPATCH:
        c->execution = c->active_handler->execute(c, &in, &out);
        if (c->execution == EXEC_FAULT)
            goto fault;
        break;
    case CTRL_UPDATE:
        break;
    }
    Control_drive(c->ports.out.control, out);
    return true;
fault:
    log_error(cycle, "Controller", "invalid instruction/state at pc=0x%05X state=%u opcode=0x%X", c->regs.pc, c->regs.state,
              c->regs.state == CTRL_DECODE ? c->decoded.opcode : c->regs.instruction.opcode);
    return false;
}
static void clock_controller(void *instance, uint64_t cycle)
{
    Controller *c = instance;
    ControllerRegisters *r = &c->regs;
    Control ctrl = Control_read(c->ports.out.control);
    static const char *names[] = {"FETCH", "RESPONSE", "DECODE", "DISPATCH", "UPDATE"};
    log_flow(cycle, "Controller", names[r->state], "pc=0x%05X opcode=%s step=%u status=%u", r->pc,
             c->active_handler ? c->active_handler->name : "-", r->step, c->execution);
    if (logging_ports_enabled())
    {
        log_block_begin("Controller");
        printf("  pc=0x%05X state=%s step=%u mem_valid=%u input_valid=%u output_ready=%u\n",
               r->pc, names[r->state], r->step, TileSignal_read(c->ports.in.memory).valid,
               StreamStatus_read(c->ports.in.stream).input_valid, StreamStatus_read(c->ports.in.stream).output_ready);
        printf("  rd=%u rd_addr=0x%05X consume=%u wr=%u wr_addr=0x%05X byte_en=%04X source=%u\n",
               ctrl.rd_en, ctrl.rd_addr, ctrl.read_consume, ctrl.wr_en, ctrl.wr_addr, ctrl.byte_enable, ctrl.write_source);
        printf("  latch=%u slot=%u field=%u compute=%u k=%u/%u N=%u V=%u accumulate=%u post=%u column=%u forward=%u target=%u input_pop=%u output_push=%u\n",
               ctrl.latch_en, ctrl.latch_slot, ctrl.latch_field, ctrl.compute_en, ctrl.k, ctrl.k_count, ctrl.neurons, ctrl.batch,
               ctrl.accumulate, ctrl.post_en, ctrl.post_col, ctrl.forward_en, ctrl.forward_slot, ctrl.input_pop, ctrl.output_push);
    }
    r->cycles++;
    if (c->execution == EXEC_WAIT)
    {
        r->bus_stall_cycles++;
        return;
    }
    switch (r->state)
    {
    case CTRL_FETCH:
        r->state = CTRL_RESPONSE;
        break;
    case CTRL_RESPONSE:
        r->ir = TileSignal_read(c->ports.in.memory).tile;
        r->state = CTRL_DECODE;
        break;
    case CTRL_DECODE:
        r->instruction = c->decoded;
        c->active_handler = c->handlers[c->decoded.opcode];
        r->next_pc = r->pc + 16;
        r->step = 0;
        r->state = CTRL_DISPATCH;
        break;
    case CTRL_DISPATCH:
        if (c->execution == EXEC_DONE)
            r->state = CTRL_UPDATE;
        else
            r->step++;
        break;
    case CTRL_UPDATE:
        if (c->active_handler->retire)
            c->active_handler->retire(c);
        r->instructions++;
        r->pc = r->next_pc;
        r->state = CTRL_FETCH;
        c->active_handler = NULL;
        break;
    }
}
const BlockOps controller_ops = {.eval = evaluate, .clock = clock_controller};
