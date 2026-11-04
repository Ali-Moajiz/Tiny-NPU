#include "controller.h"
#include "sram.h"
#include "logging.h"
#include <stdio.h>

static uint32_t read_address(const LayerConfig *cfg, TypeSel type)
{
    switch (type) {
    case TYPE_WEIGHT: return cfg->weight_base;
    case TYPE_INPUT: return cfg->input_base;
    case TYPE_BIAS: return cfg->bias_base;
    case TYPE_CONFIG: return cfg->config_base;
    }
    return SRAM_LINES;
}

static bool read_word(Controller *c, unsigned edge, unsigned layer, TypeSel type)
{
    if (edge >= SCHEDULE_CAPACITY || c->words[edge].read_enable || c->words[edge].store_en)
        return false;
    Control *word = &c->words[edge];
    word->read_enable = true;
    word->read_type = type;
    word->read_bank = layer;
    word->rd_addr = read_address(&c->layers[layer], type);
    return true;
}

static unsigned compute_layer(Controller *c, unsigned start, unsigned layer)
{
    for (unsigned k = 0; k < ARRAY_SIZE; k++) {
        Control *word = &c->words[start + k];
        word->compute_en = true;
        word->layer = word->bank = layer;
        word->k = k;
    }
    unsigned post = start + ARRAY_SIZE;
    c->words[post].post_en = true;
    c->words[post].layer = c->words[post].bank = layer;
    unsigned drain = post + POST_LATENCY;
    Control *word = &c->words[drain];
    word->layer = word->bank = layer;
    if (!layer && c->options.fuse) {
        word->forward_en = true;
        word->forward_bank = 1;
    } else {
        word->store_en = true;
        word->store_addr = c->layers[layer].output_base;
        word->byte_enable = batch_byte_enable(c->options.batch);
    }
    return drain + 1;
}

bool controller_init(Controller *c, const LayerConfig *layers, NpuOptions options)
{
    if (options.batch < 1 || options.batch > ARRAY_SIZE)
        return false;
    *c = (Controller){.options = options};
    for (unsigned layer = 0; layer < NETWORK_LAYERS; layer++)
        c->layers[layer] = layers[layer];
    for (unsigned i = 0; i < SCHEDULE_CAPACITY; i++)
        c->words[i].batch = options.batch;

    if (!read_word(c, 0, 0, TYPE_WEIGHT) || !read_word(c, 1, 0, TYPE_INPUT) ||
        !read_word(c, 2, 0, TYPE_BIAS) || !read_word(c, 3, 0, TYPE_CONFIG))
        return false;
    unsigned next = compute_layer(c, 4, 0);
    unsigned parameters = options.prefetch ? 4 : next + !options.fuse;
    if (!read_word(c, parameters, 1, TYPE_WEIGHT) ||
        !read_word(c, parameters + 1, 1, TYPE_BIAS) ||
        !read_word(c, parameters + 2, 1, TYPE_CONFIG) ||
        (!options.fuse && !read_word(c, next, 1, TYPE_INPUT)))
        return false;

    /* W/I/B come from the published register outputs. CFG may capture on the first compute edge. */
    unsigned start = options.prefetch ? next + 2u * !options.fuse : parameters + 3;
    c->length = compute_layer(c, start, 1);
    return controller_validate(c);
}

bool controller_validate(const Controller *c)
{
    if (!c->length || c->length > SCHEDULE_CAPACITY ||
        (!c->options.fuse && c->layers[0].output_base != c->layers[1].input_base))
        return false;
    bool loaded[4][PARAM_BANKS] = {{false}};
    bool hidden_stored = false;
    unsigned wave[2] = {0};
    const Control *previous = NULL;
    for (unsigned i = 0; i < c->length; i++) {
        const Control *w = &c->words[i];
        if ((w->read_enable && w->store_en) || (w->forward_en && w->store_en) ||
            w->bank >= PARAM_BANKS || w->layer >= NETWORK_LAYERS ||
            w->batch < 1 || w->batch > ARRAY_SIZE || w->batch != c->options.batch)
            return false;
        if (w->read_enable &&
            (w->read_bank >= PARAM_BANKS || !sram_address_matches_type(w->rd_addr, w->read_type) ||
             w->rd_addr != read_address(&c->layers[w->read_bank], w->read_type) ||
             (w->read_type == TYPE_INPUT && w->read_bank == 1 && !hidden_stored)))
            return false;
        if (w->compute_en) {
            if (w->bank != w->layer || w->k >= ARRAY_SIZE || w->k != wave[w->layer]++ ||
                !loaded[TYPE_WEIGHT][w->bank] || !loaded[TYPE_INPUT][w->bank] ||
                !loaded[TYPE_BIAS][w->bank])
                return false;
        }
        if (w->post_en && (!loaded[TYPE_CONFIG][w->bank] || !previous ||
                           !previous->compute_en || previous->k != ARRAY_SIZE - 1 ||
                           previous->layer != w->layer || previous->bank != w->bank))
            return false;
        if (w->store_en || w->forward_en) {
            const Control *post = POST_LATENCY ? previous : w;
            if (!post || !post->post_en || post->layer != w->layer)
                return false;
        }
        if (w->store_en) {
            if (!sram_address_matches_type(w->store_addr, TYPE_INPUT) ||
                w->store_addr != c->layers[w->layer].output_base ||
                w->byte_enable != batch_byte_enable(w->batch))
                return false;
            hidden_stored |= w->layer == 0;
        }

        /* All captures become visible on the following edge. Each file/bank
         * has a single load lifetime in this two-layer schedule. */
        if (previous && previous->read_enable) {
            TypeSel type = previous->read_type;
            unsigned bank = previous->read_bank;
            if (loaded[type][bank] ||
                (bank == w->bank && ((w->compute_en && type != TYPE_CONFIG) ||
                                     (w->post_en && type == TYPE_CONFIG))))
                return false;
            loaded[type][bank] = true;
        }
        if (w->forward_en) {
            if (w->layer != 0 || w->forward_bank != 1 || loaded[TYPE_INPUT][1])
                return false;
            loaded[TYPE_INPUT][1] = true;
        }
        previous = w;
    }
    return wave[0] == ARRAY_SIZE && wave[1] == ARRAY_SIZE &&
           previous->store_en && previous->layer == 1;
}

static void controller_publish(void *instance, uint64_t cycle)
{
    Controller *c = instance;
    Control word = c->words[c->regs.pc];
    Control *w = &word;
    ReadTag tag = c->regs.tag;
    w->capture_bank = tag.bank;
    w->we_weight = tag.valid && tag.type == TYPE_WEIGHT;
    w->we_input = tag.valid && tag.type == TYPE_INPUT;
    w->we_bias = tag.valid && tag.type == TYPE_BIAS;
    w->we_cfg = tag.valid && tag.type == TYPE_CONFIG;
    Control_drive(c->ports.out.control, word);
    log_flow(cycle,"AGU","EDGE",
             "L%u bank=%u compute=%u k=%u read=%u type=%u dest=%u addr=0x%03X capture=%u:%u%u%u%u post=%u forward=%u store=%u",
             w->layer+1,w->bank,w->compute_en,w->k,w->read_enable,w->read_type,w->read_bank,
             w->rd_addr,w->capture_bank,w->we_weight,w->we_input,w->we_bias,w->we_cfg,
             w->post_en,w->forward_en,w->store_en);
    if (w->compute_en && w->read_enable)
        log_flow(cycle,"AGU","PREFETCH","bank=%u type=%u addr=0x%03X",w->read_bank,w->read_type,w->rd_addr);
}

static void controller_clock(void *instance, uint64_t cycle)
{
    (void)cycle;
    Controller *c = instance;
    const Control *w = &c->words[c->regs.pc];
    c->regs.tag = (ReadTag){w->read_enable, w->read_type, w->read_bank};
    c->regs.done = ++c->regs.pc == c->length;
    c->regs.overlap_reads += w->compute_en && w->read_enable;
    if (logging_ports_enabled()) {
        Control out = Control_read(c->ports.out.control);
        log_block_begin("Controller");
        printf("  out layer=%u k=%u bank=%u batch=%u compute=%u post=%u forward=%u store=%u\n",
               out.layer, out.k, out.bank, out.batch, out.compute_en, out.post_en, out.forward_en, out.store_en);
        printf("  out read_en=%u rd_addr=0x%03X type=%u read_bank=%u capture_bank=%u we(W/I/B/CFG)=%u%u%u%u\n",
               out.read_enable, out.rd_addr, out.read_type, out.read_bank, out.capture_bank,
               out.we_weight, out.we_input, out.we_bias, out.we_cfg);
        printf("  regs pc=%u done=%u pending_read=%u\n", c->regs.pc, c->regs.done, c->regs.tag.valid);
    }
}

const BlockOps controller_ops = {.publish = controller_publish, .clock = controller_clock};
