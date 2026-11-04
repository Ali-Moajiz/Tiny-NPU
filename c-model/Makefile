CC ?= cc
CFLAGS ?= -std=c99 -O0 -g -Wall -Wextra -Werror
CPPFLAGS ?=
LDFLAGS ?=

# TB selects the entry point; hardware block sources are identical in both builds.
TB ?= 0
ifneq ($(words $(TB)),1)
$(error TB must contain exactly one value: 0 or 1)
endif
ifeq ($(filter $(TB),0 1),)
$(error Invalid TB=$(TB). Use 0 or 1)
endif

# Hardware dimensions remain 4x4. BATCH masks 1..4 independent vector columns.
BATCH ?= 1
PREFETCH ?= 1
FUSE ?= 1
ifneq ($(words $(BATCH)),1)
$(error BATCH must be one integer 1..4)
endif
ifeq ($(filter $(BATCH),1 2 3 4),)
$(error BATCH must be 1, 2, 3 or 4)
endif
ifneq ($(words $(PREFETCH)),1)
$(error PREFETCH must be 0 or 1)
endif
ifeq ($(filter $(PREFETCH),0 1),)
$(error PREFETCH must be 0 or 1)
endif
ifneq ($(words $(FUSE)),1)
$(error FUSE must be 0 or 1)
endif
ifeq ($(filter $(FUSE),0 1),)
$(error FUSE must be 0 or 1)
endif
# Isolate every build option; changing flags needs no clean.

LOG ?= flow
ifneq ($(words $(LOG)),1)
$(error LOG must contain exactly one mode: flow, ports, both or none)
endif
ifeq ($(filter $(LOG),flow ports both none),)
$(error Invalid LOG=$(LOG). Use flow, ports, both or none)
endif
FLOW_ENABLED := $(if $(filter flow both,$(LOG)),1,0)
PORTS_ENABLED := $(if $(filter ports both,$(LOG)),1,0)
POST_LATENCY ?= 1
LOG_FLAGS = -DPOST_LATENCY=$(POST_LATENCY) -DNPU_LOG_FLOW=$(FLOW_ENABLED) -DNPU_LOG_PORTS=$(PORTS_ENABLED) -DNPU_BATCH=$(BATCH) -DNPU_PREFETCH=$(PREFETCH) -DNPU_FUSE=$(FUSE)
BUILD_DIR := build/tb$(TB)/b$(BATCH)-p$(PREFETCH)-f$(FUSE)/post$(POST_LATENCY)/$(LOG)
TARGET := $(BUILD_DIR)/tiny_npu
CORE_SOURCES := logging.c controller.c sram.c reg128.c mac.c scale_clamp.c bus.c framework.c npu_top.c
ifeq ($(TB),1)
ENTRY_SOURCES := tb_main.c tb.c
else
ENTRY_SOURCES := main.c
endif
SOURCES := $(CORE_SOURCES) $(ENTRY_SOURCES)
OBJECTS := $(SOURCES:%.c=$(BUILD_DIR)/%.o)
LOG_FILE ?= $(LOG).log

.PHONY: all run log check clean help
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
$(BUILD_DIR)/%.o: %.c Makefile
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(LOG_FLAGS) $(CFLAGS) -MMD -MP -c $< -o $@
run: $(TARGET)
	./$(TARGET)
log: $(TARGET)
	./$(TARGET) > "$(LOG_FILE)"
	@echo "Saved $(LOG_FILE) (LOG=$(LOG))"
# End-to-end integer reference, partial batches, timing/port invariants and faults.
build/check_bus: tests/check_bus.c framework.c bus.c logging.c framework.h bus.h logging.h datapath.h Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. tests/check_bus.c framework.c bus.c logging.c -o $@
check: build/check_bus
	./build/check_bus
	POST_LATENCY=0 python3 tests/check_model.py
	POST_LATENCY=1 python3 tests/check_model.py
clean:
	$(RM) -r build
help:
	@echo 'make run TB=0 LOG=flow  # existing single run (default); no TB code linked'
	@echo 'make run TB=1 LOG=flow  # seeded random batches and cycle flow'
	@echo 'make run TB=1 LOG=none  # results only'
	@echo 'make run LOG=flow   # pipeline events (default)'
	@echo 'make run LOG=ports  # all block ports every cycle'
	@echo 'make run LOG=both   # both views'
	@echo 'make run LOG=none   # silent on success; errors still go to stderr'
	@echo 'make run TB=1 BATCH=4 LOG=flow  # four independent vectors; all 16 PEs active in compute'
	@echo 'make run BATCH=4 PREFETCH=0 FUSE=0  # disable optimizations for comparison'
	@echo 'make check          # numerical, timing, masking, bank and error regressions'
	@echo 'make log LOG=flow   # save flow.log; replace LOG for another view'
-include $(OBJECTS:.o=.d)
