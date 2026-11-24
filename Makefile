# Cross-compile lpr_pipe for the NT98539A (aarch64).
#
#   make                 # builds the link-time stubs in ai3/stubs/build, then lpr_pipe
#   make LIBS=<dir>      # link against real libraries copied off the board instead
#
# Every library is already on the board (/usr/lib); nothing from the stubs is
# shipped. The binary has plain names in NEEDED and no RPATH.

CC        = aarch64-linux-gnu-gcc
LIBS      ?= ai3/stubs/build
CFLAGS    = -O2 -Wall -Wextra -std=gnu11 -Iai3 -I.

STUB_LISTS := $(wildcard ai3/stubs/*.symbols)
STUB_SOS   := $(patsubst ai3/stubs/%.symbols,ai3/stubs/build/%,$(STUB_LISTS))
LDFLAGS   = -L$(LIBS) -Wl,--allow-shlib-undefined -Wl,--no-as-needed
LDLIBS    = -lvendor_ai3 -lvendor_ai3_pub -lhdal -lvendor_media -lvos -lprebuilt_ai -lm

lpr_pipe: src/lpr_pipe.c src/vdec_abi.h ai3/ai3_abi.h third_party/stb_image.h $(if $(filter ai3/stubs/build,$(LIBS)),$(STUB_SOS))
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@file $@ | grep -q 'ARM aarch64' || { echo "not an aarch64 binary"; exit 1; }
	@echo "built $@"

# vdec_probe: hardware JPEG decode check. Needs the 64-bit HDAL headers from the
# Novaic toolchain (Tool/release.tar.bz2, not redistributed):
#   tar -xjf release.tar.bz2 --wildcards '*/runtime64/hdal/include/*' '*/runtime64/vos/include/*'
#   make vdec_probe HDAL_RT=<extracted>/.../toolchain/repo/proprietary/runtime64
HDAL_RT ?= $(error set HDAL_RT to .../toolchain/repo/proprietary/runtime64)
vdec_probe: src/vdec_probe.c $(STUB_SOS)
	$(CC) $(CFLAGS) -I$(HDAL_RT)/hdal/include -I$(HDAL_RT)/vos/include $< -o $@ -L$(LIBS) -Wl,--allow-shlib-undefined -lhdal

# One stub per symbol list: a dummy definition for each name, with the SONAME
# the board's library has, so the linker records the right NEEDED entry.
ai3/stubs/build/%: ai3/stubs/%.symbols
	@mkdir -p $(dir $@)
	@{ echo '/* generated from $< -- do not edit */'; \
	   grep -vE '^\s*(#|$$)' $< | sed 's/.*/int &(void) { return 0; }/'; \
	 } > $@.c
	$(CC) -shared -fPIC -nostdlib -Wl,-soname,$(notdir $@) -o $@ $@.c
	@rm -f $@.c

clean:
	rm -f lpr_pipe vdec_probe
	rm -rf ai3/stubs/build

.PHONY: clean
