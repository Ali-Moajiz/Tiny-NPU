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

# Independent directories isolate BOTH testbench and logging variants.
LOG ?= flow
ifneq ($(words $(LOG)),1)
$(error LOG must contain exactly one mode: flow, ports, both or none)
endif
ifeq ($(filter $(LOG),flow ports both none),)
$(error Invalid LOG=$(LOG). Use flow, ports, both or none)
endif
FLOW_ENABLED := $(if $(filter flow both,$(LOG)),1,0)
PORTS_ENABLED := $(if $(filter ports both,$(LOG)),1,0)
LOG_FLAGS = -DNPU_LOG_FLOW=$(FLOW_ENABLED) -DNPU_LOG_PORTS=$(PORTS_ENABLED)
BUILD_DIR := build/tb$(TB)/$(LOG)
TARGET := $(BUILD_DIR)/tiny_npu
CORE_SOURCES := logging.c bus.c controller.c sram.c demux.c reg128.c mac.c activation.c scale_clamp.c fifo.c npu.c
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
# Regression: pipe the notebook reference vectors through the TB and diff against
# the expected INT8 outputs from npu_reference_vectors.json (Python integer model).
check:
	@$(MAKE) --no-print-directory TB=1 LOG=none all
	@./build/tb1/none/tiny_npu < tests/vectors.in | grep 'Predicted output' | sed 's/.*: //' > build/vectors.got
	@diff -u tests/vectors.expected build/vectors.got && echo "check: all $$(wc -l < tests/vectors.expected) reference vectors match"
clean:
	$(RM) -r build
help:
	@echo 'make run TB=0 LOG=flow  # existing single run (default); no TB code linked'
	@echo 'make run TB=1 LOG=flow  # interactive inputs, cycle flow, result, repeat'
	@echo 'make run TB=1 LOG=none  # prompts and results only'
	@echo 'make run LOG=flow   # pipeline events (default)'
	@echo 'make run LOG=ports  # all block ports every cycle'
	@echo 'make run LOG=both   # both views'
	@echo 'make run LOG=none   # silent on success; errors still go to stderr'
	@echo 'make check          # run reference vectors, diff against expected outputs'
	@echo 'make log LOG=flow   # save flow.log; replace LOG for another view'
-include $(OBJECTS:.o=.d)
