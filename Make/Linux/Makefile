MARE := ../../Util/mare/Linux/bin/mare
CONFIG ?= Debug

.PHONY: all clean

all:
	@$(MARE) config=$(CONFIG)

clean:
	@$(MARE) clean config=$(CONFIG)

%:
	@$(MARE) $@ config=$(CONFIG)
