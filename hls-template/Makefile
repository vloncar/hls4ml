SOURCE_FILES  := myproject_test.cpp firmware/myproject.cpp
HLS_CXX_FLAGS := --fpc --fp-relaxed
CXX := i++
override CXXFLAGS := $(CXXFLAGS)
VERBOSE := 1
DEVICE := Arria10

# OS-dependant tools
ifeq ($(OS),Windows_NT)
  RM  := rd /S /Q
else
  RM  := rm -rf
endif

ifeq ($(MAKECMDGOALS),)
  $(info No target specified, defaulting to test-x86-64)
  $(info Available targets: test-x86-64, test-fpga, test-gpp, clean)
endif

# Any tools installed with HLS can be found relative to the location of i++
HLS_INSTALL_DIR := $(shell which i++ | sed 's|/bin/i++||g')

# Run the i++ x86 test by default
.PHONY: default
default: myproject-x86-64

# Compile the component and testbench using g++ and run them as a regular program
.PHONY: myproject-gpp
myproject-gpp: CXX := g++
myproject-gpp: CXXFLAGS := $(CXXFLAGS) -I"$(HLS_INSTALL_DIR)/include" -L"$(HLS_INSTALL_DIR)/host/linux64/lib" -lhls_emul -o myproject-gpp
myproject-gpp: $(SOURCE_FILES)
	$(CXX) $(SOURCE_FILES) $(CXXFLAGS)
	@echo "+------------------------------------------+"
	@echo "| Run ./myproject-gpp to execute the test. |"
	@echo "+------------------------------------------+"

# Run the testbench and the component as a regular program
.PHONY: myproject-x86-64
myproject-x86-64: CXXFLAGS := $(CXXFLAGS) $(HLS_CXX_FLAGS) -I"$(HLS_INSTALL_DIR)/include" -march=x86-64 -o myproject-x86-64
myproject-x86-64: $(SOURCE_FILES)
	$(CXX) $(SOURCE_FILES) $(CXXFLAGS)
	@echo "+---------------------------------------------+"
	@echo "| Run ./myproject-x86-64 to execute the test. |"
	@echo "+---------------------------------------------+"

# Run a simulation with the C testbench and verilog component
.PHONY: myproject-fpga
ifeq ($(VERBOSE),1)
  myproject-fpga: CXXFLAGS := $(CXXFLAGS) -v
endif
myproject-fpga: CXXFLAGS := $(CXXFLAGS) $(HLS_CXX_FLAGS) -march=Arria10 -o myproject-fpga
myproject-fpga: $(SOURCE_FILES)
	$(CXX) $(SOURCE_FILES) $(CXXFLAGS)
	@echo "+-------------------------------------------+"
	@echo "| Run ./myproject-fpga to execute the test. |"
	@echo "+-------------------------------------------+"

# Clean up temprary and delivered files
CLEAN_FILES := myproject-gpp \
               myproject-gpp.prj \
               myproject-fpga \
               myproject-fpga.prj \
               myproject-x86-64 \
               myproject-x86-64.prj
clean:
	-$(RM) $(CLEAN_FILES)
