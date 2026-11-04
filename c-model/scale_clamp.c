#include "scale_clamp.h"
#include "logging.h"
/* INT32 times positive INT32 fits INT64. Quotient/remainder implements
 * nearest-even rounding without signed shifts or rounding-bias overflow. */
int8_t scale_clamp_value(int32_t value,int32_t multiplier,unsigned shift) {
    int64_t product=(int64_t)value*multiplier;
    bool negative=product<0;
    uint64_t magnitude=(uint64_t)(negative?-product:product),rounded=magnitude;
    if(shift) {
        uint64_t divisor=UINT64_C(1)<<shift;
        uint64_t quotient=magnitude/divisor,remainder=magnitude%divisor,half=divisor/2;
        if(remainder>half || (remainder==half && (quotient&1))) quotient++;
        rounded=quotient;
    }
    if(negative) return rounded>=128?INT8_MIN:(int8_t)-(int64_t)rounded;
    return rounded>127?INT8_MAX:(int8_t)rounded;
}
static bool post_valid(Control ctrl, AccSignal sums, TileSignal cfg, uint64_t cycle)
{
    if (ctrl.bank >= PARAM_BANKS || ctrl.batch < 1 || ctrl.batch > ARRAY_SIZE)
        return false;
    if (!sums.valid || !cfg.valid || !cfg.tile.lane[0] || cfg.tile.lane[0] > INT32_MAX ||
        cfg.tile.lane[1] > 62 || cfg.tile.lane[2] > 1 || cfg.tile.lane[3]) {
        log_error(cycle, "Post", "invalid final tile or configuration");
        return false;
    }
    return true;
}

static Line128 post_tile(Control ctrl, TileSums sums, Line128 cfg, uint64_t cycle)
{
    Line128 tile = {{0}};
    for (unsigned v = 0; v < ctrl.batch; v++) {
        int8_t bytes[4];
        for (unsigned r = 0; r < ARRAY_SIZE; r++) {
            int32_t x = sums.value[r][v];
            if (cfg.lane[2] && x < 0)
                x = 0;
            bytes[r] = scale_clamp_value(x, (int32_t)cfg.lane[0], cfg.lane[1]);
        }
        tile.lane[v] = pack_i8(bytes);
    }
    log_flow(cycle,"Post","POST","layer=%u valid_cycle=%llu tile=%08X_%08X_%08X_%08X",
             ctrl.layer + 1,(unsigned long long)(cycle + POST_LATENCY),
             tile.lane[3],tile.lane[2],tile.lane[1],tile.lane[0]);
    return tile;
}

static void post_clock(void *instance, uint64_t cycle)
{
    PostProcess *p = instance;
    Control c = Control_read(p->ports.in.control);
    if (logging_ports_enabled()) {
        log_block_begin("Post-process");
        if (c.post_en)
            log_tile("  in accumulators", AccSignal_read(p->ports.in.sums).tile);
        TileSignal result = TileSignal_read(p->ports.out.result);
        log_line(result.valid ? "  out result valid" : "  out result invalid", result.tile);
    }
#if POST_LATENCY == 1
    p->regs.valid = c.post_en;
    if (c.post_en) {
        p->regs.tile = post_tile(c, AccSignal_read(p->ports.in.sums).tile,
                                TileSignal_read(p->ports.in.config).tile, cycle);
        if (logging_ports_enabled())
            log_line("  regs captured", p->regs.tile);
    }
#else
    (void)cycle;
#endif
}

#if POST_LATENCY == 1
static void post_publish(void *instance, uint64_t cycle)
{
    (void)cycle;
    PostProcess *p = instance;
    TileSignal_drive(p->ports.out.result, (TileSignal){p->regs.tile, p->regs.valid});
}

static bool post_check(void *instance, uint64_t cycle)
{
    const PostProcess *p = instance;
    Control c = Control_read(p->ports.in.control);
    return !c.post_en || post_valid(c, AccSignal_read(p->ports.in.sums),
                                   TileSignal_read(p->ports.in.config), cycle);
}

const BlockOps post_ops = {.publish = post_publish, .check = post_check, .clock = post_clock};
#else
static bool post_eval(void *instance, uint64_t cycle)
{
    PostProcess *p = instance;
    Control c = Control_read(p->ports.in.control);
    TileSignal result = {0};
    if (c.post_en) {
        AccSignal sums = AccSignal_read(p->ports.in.sums);
        TileSignal cfg = TileSignal_read(p->ports.in.config);
        if (!post_valid(c, sums, cfg, cycle))
            return false;
        result = (TileSignal){post_tile(c, sums.tile, cfg.tile, cycle), true};
    }
    TileSignal_drive(p->ports.out.result, result);
    return true;
}

const BlockOps post_ops = {.eval = post_eval, .clock = post_clock};
#endif
