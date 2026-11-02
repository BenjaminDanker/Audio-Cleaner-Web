// The browser's file-type list with no browser (src/media/convert.js, mediabunny stood in for by a stub): listOutputs and what it does with a video
// that is too big for this browser to put together, and buildLimit. Usage: node tests/convert-unit.mjs
import assert from 'node:assert/strict';
import { loadConvert } from './convert-bundle.mjs';

let failed = 0;
async function check(name, fn) {
  try {
    const detail = await fn();
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} -- ${String(e.stack || e.message).split('\n').slice(0, 5).join(' / ')}`);
  }
}

const CAN = { avc: true, vp9: true, vp8: true, audio: { opus: true, aac: true } };
const MP4 = { hasVideo: true, videoCodec: 'avc', videoCanDecode: true, duration: 600, channels: 2, width: 1920, height: 1080 };
const SOUND = { hasVideo: false, duration: 600, channels: 2 };
const AVI = { hasVideo: true, viaLibav: true, videoCodec: null, duration: 600, channels: 2 };
const byId = (list) => Object.fromEntries(list.map((o) => [o.id, o]));
const videos = (list) => list.filter((o) => o.kind === 'video');

await check('a video this browser can put together is offered every video type as made here', async () => {
  const m = await loadConvert(CAN);
  const list = await m.listOutputs({ info: MP4 });
  assert.deepEqual(videos(list).map((o) => [o.id, !!o.server]), [['mp4', false], ['mov', false], ['mkv', false], ['webm', false]]);
  assert.deepEqual(list.filter((o) => o.kind === 'audio').map((o) => o.id), ['wav', 'flac', 'mp3', 'ogg', 'm4a']);
});

await check('tooBig: every video type of a video is made on our side, the sound types and the other sources are as they were', async () => {
  const m = await loadConvert(CAN);
  const big = await m.listOutputs({ info: MP4, tooBig: true });
  assert.deepEqual(videos(big).map((o) => [o.id, o.server === true, o.ok]), [['mp4', true, true], ['mov', true, true], ['mkv', true, true], ['webm', true, true]]);
  const small = await m.listOutputs({ info: MP4, tooBig: false });
  assert.deepEqual(big.filter((o) => o.kind === 'audio'), small.filter((o) => o.kind === 'audio'));
  for (const info of [SOUND, AVI]) { // (a sound has no picture to put together; a libav video is made on our side anyway)
    assert.deepEqual(await m.listOutputs({ info, tooBig: true }), await m.listOutputs({ info, tooBig: false }), JSON.stringify(info));
  }
  assert.deepEqual(videos(await m.listOutputs({ info: AVI })).map((o) => o.server), [true, true, true, true]);
  assert.deepEqual(videos(await m.listOutputs({ info: SOUND })).map((o) => !!o.server), [false, false, false, false]);
  assert.deepEqual(await m.listOutputs({ info: MP4 }), await m.listOutputs({ info: MP4, serverAudio: false, tooBig: false })); // the default is not too big
});

await check('serverAudio and tooBig together: the sound types are all offered (our side writes them), the video types are still ours', async () => {
  const m = await loadConvert({ avc: true, vp9: false, vp8: false, audio: {} });
  const list = byId(await m.listOutputs({ info: MP4, serverAudio: true, tooBig: true }));
  assert.ok(list.mp3.ok && list.ogg.ok && list.m4a.ok && list.flac.ok);
  assert.ok(['mp4', 'mov', 'mkv', 'webm'].every((id) => list[id].server === true));
});

await check('a browser that cannot encode a picture still copies it into a type that takes it, and makes the rest on our side', async () => {
  const m = await loadConvert({ avc: false, vp9: false, vp8: false, audio: { opus: true, aac: true } });
  const list = byId(await m.listOutputs({ info: MP4 }));
  assert.deepEqual(['mp4', 'mov', 'mkv', 'webm'].map((id) => !!list[id].server), [false, false, false, true]);
  const big = byId(await m.listOutputs({ info: MP4, tooBig: true }));
  assert.ok(['mp4', 'mov', 'mkv', 'webm'].every((id) => big[id].server === true));
});

/** Runs fn with a Worker in the global scope (a browser has one; node has none), the thing outputCaps() needs, besides WebAssembly, to count on the WebAssembly AAC and MP3 encoders. */
async function withWorker(fn) {
  const had = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  globalThis.Worker = class {};
  try {
    return await fn();
  } finally {
    if (had) Object.defineProperty(globalThis, 'Worker', had); else delete globalThis.Worker;
  }
}
const MPEG4 = { hasVideo: true, videoCodec: 'mpeg4', videoCanDecode: true, duration: 600, channels: 2, width: 1920, height: 1080 }; // (a picture no file type here holds as it is: it is made again)

await check('AAC without the browser\'s own encoder: with WebAssembly and a Worker it is still written here (M4A, and an MP4 made in the browser); ensureAac registers the WebAssembly encoder once', async () => {
  delete globalThis.__AAC_REGISTERED;
  const m = await withWorker(() => loadConvert({ avc: true, vp9: true, vp8: true, audio: { opus: true, aac: false } }));
  assert.equal((await m.outputCaps()).audio.aac, true);
  const list = byId(await m.listOutputs({ info: MP4 }));
  assert.ok(list.m4a.ok, 'M4A is offered');
  assert.deepEqual(['mp4', 'mov', 'mkv', 'webm'].map((id) => !!list[id].server), [false, false, false, false]);
  assert.deepEqual(byId(await m.listOutputs({ info: SOUND })).mp4.server ?? false, false); // (a sound into an MP4: black picture made here)
  assert.equal(globalThis.__AAC_REGISTERED, undefined, 'nothing is downloaded or registered before AAC is needed');
  assert.equal(await withWorker(() => m.ensureAac()), true);
  assert.equal(await withWorker(() => m.ensureAac()), true);
  assert.equal(globalThis.__AAC_REGISTERED, 1, 'registered once');
});

await check('no WebAssembly Worker and no AAC of its own (Opus only): M4A is not offered, and an MP4 is made on our side rather than with Opus sound; Ogg stays', async () => {
  const m = await loadConvert({ avc: true, vp9: true, vp8: true, audio: { opus: true, aac: false } }); // (node has no Worker: the same as a browser without one)
  assert.equal((await m.outputCaps()).audio.aac, false);
  const list = byId(await m.listOutputs({ info: MP4 }));
  assert.equal(list.m4a.ok, false);
  assert.ok(list.ogg.ok);
  assert.deepEqual(['mp4', 'mov', 'mkv', 'webm'].map((id) => !!list[id].server), [true, false, false, false]); // (MOV can still hold PCM sound, MKV and WebM hold Opus)
});

await check('a browser that cannot write H.264: an MP4 (or MOV) with a new picture size is made on our side, never with VP9 inside; MKV and WebM may still use VP9', async () => {
  const m = await withWorker(() => loadConvert({ avc: false, vp9: true, vp8: true, audio: { opus: true, aac: false } }));
  // a picture no type holds as it is is made again: MP4 and MOV could only get H.264, so they are ours; MKV and WebM make it with VP9 here
  const list = byId(await m.listOutputs({ info: MPEG4 }));
  assert.deepEqual(['mp4', 'mov', 'mkv', 'webm'].map((id) => !!list[id].server), [true, true, false, false]);
  assert.deepEqual(await m.videoPlan(MPEG4, 'mkv'), { ok: true, reencode: true, codec: 'vp9', why: '' });
  assert.equal((await m.videoPlan(MPEG4, 'mp4')).ok, false);
  // the same file made smaller
  for (const info of [MP4, MPEG4]) {
    const sizes = Object.fromEntries(['mp4', 'mov', 'mkv', 'webm'].map((id) => [id, null]));
    for (const id of Object.keys(sizes)) sizes[id] = await m.listSizes(info, id);
    for (const id of ['mp4', 'mov']) assert.ok(sizes[id].filter((x) => x.px).every((x) => x.server === true), `${id} sizes: ${JSON.stringify(sizes[id])}`);
    for (const id of ['mkv', 'webm']) assert.ok(sizes[id].every((x) => x.ok && !x.server), `${id} sizes: ${JSON.stringify(sizes[id])}`);
    assert.equal((await m.videoPlan(info, 'mkv', 720, { width: 1280, height: 720 })).codec, 'vp9');
    assert.equal((await m.videoPlan(info, 'mp4', 720, { width: 1280, height: 720 })).ok, false);
  }
  // copying the picture as it is stays as it was: an H.264 video into an MP4, MOV or MKV is made here (WebM makes it again with VP9), and the size 'same' of an MP4 is not ours
  const own = byId(await m.listOutputs({ info: MP4 }));
  assert.deepEqual(['mp4', 'mov', 'mkv', 'webm'].map((id) => !!own[id].server), [false, false, false, false]);
  assert.equal((await m.listSizes(MP4, 'mp4')).find((x) => !x.px).server, undefined);
});

await check('a browser that can write H.264: an MP4 with a new picture size is made here with H.264, and VP9 is never the picture of an MP4', async () => {
  const m = await withWorker(() => loadConvert({ avc: true, vp9: true, vp8: true, audio: { opus: true, aac: false } }));
  const mp4 = await m.videoPlan(MP4, 'mp4', 720, { width: 1280, height: 720 });
  assert.deepEqual(mp4, { ok: true, reencode: true, codec: 'avc', why: '' });
  assert.ok((await m.listSizes(MP4, 'mp4')).every((x) => x.ok && !x.server));
  const odd = await m.videoPlan(MPEG4, 'mp4'); // (a picture an MP4 does not hold is made again as H.264)
  assert.equal(odd.codec, 'avc');
});

await check('buildLimit: none where the browser can write to disk, 1.5 GB where it can not', async () => {
  const m = await loadConvert(CAN);
  const had = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const hadHandle = globalThis.FileSystemFileHandle;
  try {
    Object.defineProperty(globalThis, 'navigator', { value: undefined, configurable: true, writable: true });
    assert.equal(m.buildLimit(), 1.5 * 1024 ** 3);
    Object.defineProperty(globalThis, 'navigator', { value: { storage: { getDirectory() {} } }, configurable: true, writable: true });
    assert.equal(m.buildLimit(), 1.5 * 1024 ** 3); // (no writable file handles)
    globalThis.FileSystemFileHandle = class { createWritable() {} };
    assert.equal(m.buildLimit(), null);
    Object.defineProperty(globalThis, 'navigator', { value: { storage: {} }, configurable: true, writable: true });
    assert.equal(m.buildLimit(), 1.5 * 1024 ** 3);
  } finally {
    if (had) Object.defineProperty(globalThis, 'navigator', had); else delete globalThis.navigator;
    if (hadHandle) globalThis.FileSystemFileHandle = hadHandle; else delete globalThis.FileSystemFileHandle;
  }
});

console.log(failed ? `\n${failed} check(s) failed` : '\nall passed');
process.exit(failed ? 1 : 0);
