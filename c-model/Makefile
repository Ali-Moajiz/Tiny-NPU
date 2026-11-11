CC ?= cc
TB ?= 0
BATCH ?= 1
LOG ?= flow
POST_LATENCY ?= 1
SANITIZE ?= 0
ifneq ($(filter $(TB),0 1),$(TB))
$(error TB must be 0 or 1)
endif
ifneq ($(filter $(BATCH),1 2 3 4),$(BATCH))
$(error BATCH must be 1..4 and match the compiled program)
endif
ifneq ($(filter $(POST_LATENCY),0 1),$(POST_LATENCY))
$(error POST_LATENCY must be 0 or 1)
endif
ifneq ($(filter $(LOG),none flow ports both),$(LOG))
$(error LOG must be none, flow, ports or both)
endif
ifneq ($(origin PREFETCH),undefined)
$(error PREFETCH is encoded by the instruction program, not a C-model build switch)
endif
ifneq ($(origin FUSE),undefined)
$(error FUSE is encoded by LOAD_OPERAND/FORWARD instructions, not a C-model build switch)
endif
BUILD := build/tb$(TB)-b$(BATCH)-post$(POST_LATENCY)-$(LOG)-san$(SANITIZE)
BIN := $(BUILD)/tiny_npu
SOURCES := bus.c framework.c logging.c isa.c controller.c instructions.c sram.c stream.c mac.c activation.c scale_clamp.c npu_top.c main.c
ifeq ($(TB),1)
SOURCES += tb.c
endif
OBJECTS := $(SOURCES:%.c=$(BUILD)/%.o)
CPPFLAGS += -DNPU_TB=$(TB) -DNPU_BATCH=$(BATCH) -DPOST_LATENCY=$(POST_LATENCY) -DNPU_LOG_FLOW=$(if $(filter flow both,$(LOG)),1,0) -DNPU_LOG_PORTS=$(if $(filter ports both,$(LOG)),1,0)
CFLAGS ?= -O2 -g
CFLAGS += -std=c99 -Wall -Wextra -Werror -MMD -MP
ifeq ($(SANITIZE),1)
CFLAGS += -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie
LDFLAGS += -fsanitize=address,undefined -no-pie
endif
.PHONY: all run clean test
all: $(BIN)
$(BIN): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
run: $(BIN)
	$(BIN) --sram "$(SRAM)" $(ARGS)
test: $(BIN)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. tests/check_bus.c bus.c framework.c logging.c $(LDFLAGS) -o $(BUILD)/check_bus
	$(BUILD)/check_bus
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. tests/check_core.c $(filter-out main.c tb.c,$(SOURCES)) $(LDFLAGS) -o $(BUILD)/check_core
	$(BUILD)/check_core
	python3 tests/check_model.py --binary $(BIN) --post-latency $(POST_LATENCY)
clean:
	rm -rf build
-include $(OBJECTS:.o=.d)
