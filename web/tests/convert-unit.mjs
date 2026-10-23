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
