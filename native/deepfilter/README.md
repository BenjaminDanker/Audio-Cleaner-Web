# DeepFilterNet libDF (tract 0.23) - native + WebAssembly builds

Reproducible builds of [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
`libDF` (pinned to commit `d375b2d`) for

* a native shared library (C API, loaded from Python with `ctypes`): `libdf.so` on Linux,
  `libdf.dylib` on macOS, `df.dll` on Windows (the names `acw_common.dfn.LIB_NAME` expects), and
* a WebAssembly module `df.js` + `df_bg.wasm` (wasm-bindgen `web` target, SIMD128) for browsers.

Neither embeds the DFN3 model: pass `DeepFilterNet3_onnx.tar.gz` at runtime
(a file path natively, the tarball bytes in the browser).

## What `tract-0.23.patch` changes

* **tract 0.21.4 -> 0.23.8** (`tract-core/onnx/pulse/hir`, pinned `=0.23.8`, the latest 0.23.x) and
  the resulting API port in `libDF/src/tract.rs` (`tract_core::ndarray`, plain array views,
  `TValue` = `Arc<Tensor>`, `SimpleState::new(&plan)`, `with_outputs_by_name`). tract 0.23 has
  SIMD kernels for wasm (0.21 had none), output matches the 0.21 build to ~130 dB SNR.
* **Quality fix**: both enhancement stages (ERB gains + deep filter) run for every frame.
  libDF's default local-SNR thresholds (-10/30/20 dB; -15/35/35 in the old capi) skip stages
  frame by frame, which costs a lot of quality (WorkAdventure measured PESQ 2.12 vs 3.22 on
  VoiceBank+DEMAND). `df_create` now uses thresholds `-inf, +inf, +inf`.
* `features capi` / `wasm` no longer force `default-model`; `wasm` also enables the browser
  backend of getrandom 0.4 (pulled in by tract 0.23).
* **capi.rs** (all old exports kept): `df_create` (always-on stages, post filter off),
  new `df_create_with_params(...)` to choose thresholds/post filter, new `df_last_error()`.
  `df_create*` return `NULL` (not a panic) if the model cannot be loaded or an argument is bad;
  `df_process_frame` catches panics and returns NaN (input copied to output). Catching needs
  unwinding, so the native build uses the new cargo profile `release-capi`
  (= `release-lto` with `panic = "unwind"`; `release-lto` aborts on panic).
* **wasm.rs**: `df_create` (always-on stages) and new `df_create_with_params(model_bytes,
  atten_lim, min_db_thresh, max_db_erb_thresh, max_db_df_thresh, post_filter_beta)`, new batch
  `df_process_frames(st, input)` (N*hop samples in one call, returns a `Float32Array`), new
  `df_free`, new `df_reset(st)`; failures are thrown as JS `Error`s; `console_error_panic_hook`
  is installed.
* **tract.rs / capi.rs / wasm.rs**: `df_reset(st)` returns an instance to its just-created state
  (the three networks' tract states, rolling spectrum buffers, STFT memories, running
  normalisation, skip counter; attenuation/thresholds/post filter are kept) so one instance can be
  reused for many unrelated signals with bit-identical output to a fresh one: 0.02 ms instead of
  ~39 ms for `df_create` (model re-read and hashed).

`Cargo.lock` here is the pinned dependency set used by the build (`cargo build --locked`).

## Build

```sh
native/deepfilter/build.sh            # native + wasm
native/deepfilter/build.sh native     # only the native library (libdf.so / libdf.dylib / df.dll)
native/deepfilter/build.sh wasm       # only df.js + df_bg.wasm
BUILD_DIR=... OUT=... native/deepfilter/build.sh
```

Defaults: `BUILD_DIR=~/audio-cleaner/src/deepfilter-build` (clone + `target/`),
`OUT=$BUILD_DIR/out` (`libdf.so`, `wasm/{df.js,df_bg.wasm,*.d.ts,LICENSE-*}`, `SHA256SUMS`).

### Other platforms (desktop app)

The `native` target also runs on macOS and on Windows (Git Bash with the MSVC Rust toolchain);
`.github/workflows/desktop.yml` builds it that way for the desktop installers.

* The library is installed under the platform's name (`libdf.dylib`, `df.dll`) taken from
  `target/release-capi/`; the checksum file uses `shasum -a 256` if `sha256sum` is missing.
* `CARGO_BUILD_TARGET=<triple>` (e.g. `x86_64-apple-darwin` on an arm64 Mac) cross-compiles; the
  script adds the rustup target if needed, reads the output from
  `target/<triple>/release-capi/` and names the library after the triple.
* On macOS `MACOSX_DEPLOYMENT_TARGET` defaults to `11.0`.
* The clone is made with `core.autocrlf=false`, otherwise a CRLF checkout on Windows makes
  `tract-0.23.patch` fail to apply (the repo's `.gitattributes` pins the patch, `Cargo.lock` and the
  shell scripts to LF for the same reason).
* Only Linux x86_64 is covered by the tests in this directory; the macOS and Windows builds are
  produced and smoke-tested (`--selftest` of the frozen app) by CI.
Needs rustup/cargo, git, network on the first run. The script adds the
`wasm32-unknown-unknown` target and installs the pinned prebuilt `wasm-pack` 0.15.0 into
`~/.cargo/bin` (sha256-checked) if missing; wasm-pack fetches wasm-bindgen and wasm-opt itself.
Native is built with `cargo build --locked --profile release-capi -p deep_filter
--no-default-features --features capi --lib`, wasm with
`RUSTFLAGS="-C target-feature=+simd128" wasm-pack build libDF --target web --release
--no-default-features --features wasm -- --locked`. A fresh build takes ~2 min (native) +
~5 min (wasm, mostly wasm-opt) on a Ryzen 7 1700X.

## Tests

```sh
~/audio-cleaner/venv/bin/python native/deepfilter/test_native.py $OUT/libdf.so \
    ~/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz --save /tmp/sig
node native/deepfilter/test_wasm.mjs $OUT/wasm ~/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz \
    --input /tmp/sig.in.f32 --ref /tmp/sig.out.f32      # Node >= 18
```

## C API (`libdf.so`)

```c
DFState *df_create(const char *path, float atten_lim, const char *log_level);
DFState *df_create_with_params(const char *path, float atten_lim, float min_db_thresh,
                               float max_db_erb_thresh, float max_db_df_thresh,
                               float post_filter_beta, const char *log_level);
const char *df_last_error(void);               /* thread-local, do not free; NULL if none */
size_t df_get_frame_length(DFState *st);       /* 480 samples @ 48 kHz */
float  df_process_frame(DFState *st, float *input, float *output); /* lsnr, NaN on error */
void   df_set_atten_lim(DFState *st, float lim_db);
void   df_set_post_filter_beta(DFState *st, float beta);
char  *df_next_log_msg(DFState *st);  void df_free_log_msg(char *msg);
float  df_process_frame_raw(DFState *st, float *input, float **gains, float **coefs);
int    df_reset(DFState *st);          /* back to the just-created state; 0 ok, -1 error */
void   df_free(DFState *st);
```

Always-on stages: `df_create_with_params(path, atten, -INFINITY, INFINITY, INFINITY, beta, NULL)`.

## Licences

* DeepFilterNet (libDF): MIT OR Apache-2.0 (see the `LICENSE-*` files shipped next to the wasm).
* tract: MIT OR Apache-2.0. Other crates linked in are permissively licensed
  (`cargo tree`/`cargo license` in the build dir for the full list).
* The patch is our own; WorkAdventure's MIT-licensed
  [noise-suppression](https://github.com/workadventure/noise-suppression) build script was used
  as a reference for the approach only (nothing vendored).

## Measured (Ryzen 7 1700X, 20 s of synthetic noisy speech at 48 kHz, 2000 frames, 1 thread)

| | old (tract 0.21.4, thresholds -15/35/35) | new (tract 0.23.8, always-on stages) |
|---|---|---|
| native, ms/frame (10 ms frame) | 1.03 - 1.06 (RTF 0.105) | **0.88 - 0.95** (RTF 0.09) |
| native, model load | 415 ms | 115 ms |
| native `libdf.so` size | 15.4 MB (8 MB model embedded) | 25.4 MB (no model) |
| wasm in Node 18 (V8 10.2, SIMD), ms/frame | - | 1.1 - 1.4 (RTF ~0.12); `df_process_frames` batch == per-frame in Node |
| wasm compile+instantiate / model load in Node | - | ~0.2 s / ~0.27 s |
| wasm size | - | 14.0 MB raw, 3.88 MB gzip (3.87 MB `-9`), 2.56 MB zstd -19 |

Native output of the new library matches the old one to 130 dB SNR (max abs diff 5e-7) on frames
where the old thresholds are inactive, and the wasm output matches native to 129 dB, so the
tract port is numerically transparent. Timings vary +-30 % when other jobs run on the box (the
ranges above are quiet-machine runs); the wasm figure is Node, browser numbers will differ.
Both artifacts rebuild bit-identically (same sha256 over two builds, see `SHA256SUMS`).
