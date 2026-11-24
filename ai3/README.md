# AI3 / HDAL link-time pieces (from onnx2novaic-98539/board, branch feat-input)

- `ai3_abi.h`: AI3 and HDAL declarations written from Novatek's AI3 / NPU
  Runtime programming guides. These are declarations only, not Novatek headers.
- `stubs/*.symbols`: the SONAME and exported names of each board library.
  `make` turns every list into a dummy shared object, so the cross-build needs
  nothing copied off the device.

Nothing here is shipped to the board. The real libraries are already in
`/usr/lib` there, and `lpr_pipe` has plain library names in `NEEDED` and no
`RPATH`.
