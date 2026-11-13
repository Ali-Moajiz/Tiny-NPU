#include "stream.h"
#include "logging.h"
#include <stdio.h>
bool stream_load_binary(Stream *s, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return false;
    unsigned char bytes[16];
    size_t n;
    s->regs.input_count = 0;
    s->regs.input_index = 0;
    bool ok = true;
    while ((n = fread(bytes, 1, 16, f)) != 0)
    {
        if (n != 16 || s->regs.input_count == STREAM_CAPACITY)
        {
            ok = false;
            break;
        }
        Line128 line = {{0}};
        for (unsigned i = 0; i < 16; i++)
            line_set_byte(&line, i, bytes[i]);
        s->regs.input[s->regs.input_count++] = line;
    }
    if (ferror(f))
        ok = false;
    fclose(f);
    return ok;
}
bool stream_save_binary(const Stream *s, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return false;
    bool ok = true;
    for (size_t b = 0; b < s->regs.output_count; b++)
        for (unsigned i = 0; i < 16; i++)
            if (fputc((uint8_t)line_byte(s->regs.output[b], i), f) == EOF)
                ok = false;
    if (fclose(f))
        ok = false;
    return ok;
}
static void publish(void *instance, uint64_t cycle)
{
    Stream *s = instance;
    bool valid = s->regs.input_index < s->regs.input_count && cycle >= s->regs.input_available_cycle;
    TileSignal data = {0};
    data.valid = valid;
    if (valid)
        data.tile = s->regs.input[s->regs.input_index];
    TileSignal_drive(s->ports.out.input, data);
    StreamStatus_drive(s->ports.out.status, (StreamStatus){valid, s->regs.output_enabled && s->regs.output_count < STREAM_CAPACITY && cycle >= s->regs.output_available_cycle});
}
static bool check(void *instance, uint64_t cycle)
{
    Stream *s = instance;
    Control c = Control_read(s->ports.in.control);
    StreamStatus status = StreamStatus_read(s->ports.out.status);
    if ((c.input_pop && !status.input_valid) || (c.output_push && (!status.output_ready || !TileSignal_read(s->ports.in.response).valid)))
    {
        log_error(cycle, "Stream", "invalid stream handshake");
        return false;
    }
    return true;
}
static void clock_stream(void *instance, uint64_t cycle)
{
    Stream *s = instance;
    Control c = Control_read(s->ports.in.control);
    if (logging_ports_enabled())
    {
        log_block_begin("Stream endpoint");
        StreamStatus st = StreamStatus_read(s->ports.out.status);
        printf("  input_valid=%u output_ready=%u input_pop=%u output_push=%u response_valid=%u\n", st.input_valid, st.output_ready, c.input_pop, c.output_push, TileSignal_read(s->ports.in.response).valid);
        log_line("input_data", TileSignal_read(s->ports.out.input).tile);
        log_line("output_data", TileSignal_read(s->ports.in.response).tile);
    }
    if (c.input_pop)
    {
        s->regs.input_index++;
        log_flow(cycle, "Stream", "INPUT", "beat=%zu", s->regs.input_index - 1);
    }
    if (c.output_push)
    {
        s->regs.output[s->regs.output_count++] = TileSignal_read(s->ports.in.response).tile;
        log_flow(cycle, "Stream", "OUTPUT", "beat=%zu", s->regs.output_count - 1);
    }
}
const BlockOps stream_ops = {.publish = publish, .check = check, .clock = clock_stream};
