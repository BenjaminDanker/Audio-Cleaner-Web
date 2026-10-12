# Third-party licences

The Audio Cleaner OBS plugin is licensed under GPL-2.0-or-later (see `LICENSE`).
It contains or links the components below. Each is fetched at build time as a
release archive pinned by SHA-256 (see `core/cmake/third_party.cmake`); nothing
is vendored in this repository.

| Component | Version | Licence | Used for |
|---|---|---|---|
| [libobs / obs-frontend-api](https://github.com/obsproject/obs-studio) | 31.1.1 (build target) | GPL-2.0-or-later | the host API (dynamically linked, not shipped) |
| [obs-plugintemplate](https://github.com/obsproject/obs-plugintemplate) | 3e7d7ac | GPL-2.0 | CMake scaffolding in `cmake/`, `CMakePresets.json`, `src/plugin-support.*` |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.11.3 | MIT | JSON for protocol messages (header only) |
| [IXWebSocket](https://github.com/machinezone/IXWebSocket) | 11.4.6 | BSD-3-Clause | WebSocket client (sources compiled into the module) |
| [libopus](https://opus-codec.org/) | 1.5.2 | BSD-3-Clause | Opus codec for the audio stream |
| [Mbed TLS](https://github.com/Mbed-TLS/mbedtls) | 3.6.7 | Apache-2.0 OR GPL-2.0-or-later | TLS on Windows (and on Linux builds without OpenSSL headers) |
| OpenSSL | 3.x (system) | Apache-2.0 | TLS on Linux (dynamically linked system library) |
| Apple Security.framework (SecureTransport) | system | - | TLS on macOS |

Because Apache-2.0 code (OpenSSL 3, and Mbed TLS if the Apache option were taken) is
not compatible with GPL-2.0-only, binaries that link it are distributed under the
"or later" option, i.e. under GPL-3.0 terms. Mbed TLS is used here under its
GPL-2.0-or-later option. The system OpenSSL on Linux is a system library.

The operating-system TLS stacks and the OBS libraries are not redistributed with the module.

---

## nlohmann/json 3.11.3 (MIT)

```
MIT License 

Copyright (c) 2013-2022 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## IXWebSocket 11.4.6 (BSD-3-Clause)

```
Copyright (c) 2018 Machine Zone, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## libopus 1.5.2 (BSD-3-Clause)

```
Copyright 2001-2023 Xiph.Org, Skype Limited, Octasic,
                    Jean-Marc Valin, Timothy B. Terriberry,
                    CSIRO, Gregory Maxwell, Mark Borgerding,
                    Erik de Castro Lopo, Mozilla, Amazon

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Opus is subject to the royalty-free patent licenses which are
specified at:

Xiph.Org Foundation:
https://datatracker.ietf.org/ipr/1524/

Microsoft Corporation:
https://datatracker.ietf.org/ipr/1914/

Broadcom Corporation:
https://datatracker.ietf.org/ipr/1526/
```

## Mbed TLS 3.6.7 (Apache-2.0 OR GPL-2.0-or-later)

Mbed TLS files are provided under a dual Apache-2.0 OR GPL-2.0-or-later licence;
this project takes the GPL-2.0-or-later option. Licence texts and the copyright
notices of the Mbed TLS contributors are in the upstream tarball
(`LICENSE`, `tests/`, `library/`), https://github.com/Mbed-TLS/mbedtls/blob/mbedtls-3.6.7/LICENSE

## OpenSSL 3.x (Apache-2.0)

Used as a dynamically linked system library on Linux. https://www.openssl.org/source/license.html

## obs-plugintemplate (GPL-2.0)

The `cmake/` modules, the CMake presets, `src/plugin-support.*` and the `buildspec.json`
mechanism are adapted from https://github.com/obsproject/obs-plugintemplate
(GPL-2.0, Copyright the OBS Project contributors). Dependency hashes for OBS 31.1.1
and obs-deps 2025-07-11 are copied from that template.
