import { Link } from 'react-router';
import { Code, DevPage, Section, Table, apiBase } from '../components/DevDocs.jsx';
import { usePageTitle } from '../hooks.js';

// Written by hand from docs/streaming-protocol.md: when the protocol changes, change both (and docs/api.md if it is
// about keys or billing). Plain words, no internals; text inside code samples is exempt from the copy test.

const TOC = [
  ['connect', 'Connect'],
  ['start', '1. Start'],
  ['errors', 'Start errors'],
  ['frames', '2. Audio frames'],
  ['events', '3. Events from us'],
  ['translations', 'Translations'],
  ['control', '4. Messages to us'],
  ['echo', '5. Echo cancellation'],
  ['client', 'Building a client'],
];

export default function DevelopersLive() {
  usePageTitle('Live protocol');
  const url = apiBase().replace(/^http/, 'ws') + '/api/stream/ws';
  return (
    <DevPage
      title="Live protocol"
      lead="Clean a microphone and write live captions over one WebSocket. This is the whole protocol the OBS plugin speaks, so you can build your own client."
      toc={TOC}
    >
      <Section id="connect" title="Connect">
        <p>Open one WebSocket for each live source:</p>
        <Code label="WebSocket address">{url}</Code>
        <p>Authorise it with <code>Authorization: Bearer acw_...</code>, using a key that has the <code>stream</code> scope (create one on <Link to="/app/live">Live &amp; keys</Link>; give a plugin a key with only that scope). Web pages on this site can use the sign-in cookie instead.</p>
        <p>If we refuse the connection, you get a plain HTTP status instead of a WebSocket:</p>
        <Table
          head={['Status', 'Meaning']}
          rows={[
            ['401', 'No credentials, or the key is wrong or revoked, or the browser sign-in is gone.'],
            ['403', 'The account is suspended, or the key doesn’t have the stream scope.'],
            ['429', 'Too many connection attempts, or too many live sessions started this hour (60 for each account, 90 for each network; failed starts count). Retry-After says how many seconds to wait.'],
          ]}
        />
        <p>See <Link to="/developers">For developers</Link> for keys, prices and spending limits. The price per minute comes back in <code>ready</code> as <code>usdPerMinute</code>; about five minutes are set aside when a session starts, more is added a minute at a time, and at the end you pay for the seconds of audio that were processed.</p>
      </Section>

      <Section id="start" title="1. Start">
        <p>Send one text message within 15 seconds:</p>
        <Code label="Start message">{`{"type": "start", "version": 1,
 "denoise": "light" | null,
 "captions": "local" | "premium" | null,
 "language": "en" | ["en", "es"] | null,
 "translate": ["es", "de"],
 "sampleRate": 48000,
 "codec": "opus" | "pcm_s16le",
 "opusBitrate": 96000,
 "attenLimDb": 100,
 "echoReference": true,
 "client": "my-client/0.1.0"}`}</Code>
        <ul>
          <li>Set at least one of <code>denoise</code> and <code>captions</code>. <code>"local"</code> captions are Standard (English only); <code>"premium"</code> captions are extra accurate and work in other languages.</li>
          <li><code>language</code> is the spoken language. If someone switches between languages, give a list of up to three: extra accurate captions follow the switches. Standard captions need <code>"en"</code> or null.</li>
          <li><code>sampleRate</code> must be 48000. <code>opusBitrate</code> is 24000 to 160000 (we send Opus back at that rate). <code>attenLimDb</code> is how much noise may be removed, in dB.</li>
          <li><code>echoReference</code>, <code>translate</code> and the rest are optional; see below.</li>
        </ul>
        <p><b>Free live time.</b> Every account gets two hours of Standard live once. A session that only uses <code>denoise: "light"</code> and/or <code>captions: "local"</code> runs on those seconds while there are any: nothing is set aside or charged, and the seconds come off the free time. Extra accurate captions and translations are always paid. One free session can run at a time, and free sessions wait while live is busy. The <code>ready</code> message of a free session has <code>"trial": true</code> and <code>trialSecondsLeft</code>.</p>
        <p>We answer with:</p>
        <Code label="Ready message">{`{"type": "ready", "sessionId": "...", "sampleRate": 48000, "hopSamples": 480,
 "modelLatencyMs": 39.0, "usdPerMinute": 0.01, "maxSessionSeconds": 43200,
 "echoReference": true}`}</Code>
        <p><code>echoReference</code> is only there (true) when you asked for it and we will use it. <code>modelLatencyMs</code> is the delay we add: the same 39 ms in every session, whatever is switched on, so you can change the settings while live without your timing changing. Or we answer <code>{'{"type":"error","code":...,"message":...}'}</code> and close.</p>
        <p>When the setup you asked for doesn&rsquo;t fit right now (live has a measured capacity), we refuse and list what would still work:</p>
        <Code label="A full setup">{`{"type": "error", "code": "full", "available": [{"captions": "premium"}],
 "message": "Live noise removal is full right now. Still available: extra-accurate captions. Try again in a few minutes."}`}</Code>
        <p>Extra accurate captions on their own are never refused for capacity.</p>
      </Section>

      <Section id="errors" title="Start errors">
        <p>Every error may carry <code>retryAfterSeconds</code>: don&rsquo;t try again sooner than that.</p>
        <Table
          head={['Code', 'Meaning']}
          rows={[
            ['bad_start', 'No valid start message within 15 seconds.'],
            ['bad_config', 'A setting we can’t use: not 48 kHz, a language code that isn’t 2 or 3 letters, a noise strength that isn’t a number, or Standard captions with a language other than English.'],
            ['bad_tier', 'An option we don’t offer.'],
            ['nothing_enabled', 'Neither denoise nor captions is set.'],
            ['insufficient_funds', 'The credit set aside up front (5 minutes) is more than the available balance.'],
            ['key_spending_cap', 'The key has a monthly spending limit and the 5 minutes set aside would go over it. The message says how much is left and when the month starts over.'],
            ['tos', 'The current Terms haven’t been accepted.'],
            ['trial_busy', 'A free session was turned away because live is busy. Paid sessions aren’t affected. Retry after 120 seconds.'],
            ['too_many_sessions', 'Four live sessions for each account (six for each network) are already running, or a second free session was asked for. A session of yours that has been silent for 10 seconds is taken to be dropped and is replaced instead. Retry after 30 seconds.'],
            ['full', 'Live capacity is used up right now; "available" lists what still fits. Retry after 60 seconds.'],
            ['restarting', 'The service is being updated; new sessions are refused for a minute or so. Retry after 60 seconds.'],
            ['unavailable', 'Live processing is busy or offline (or extra accurate captions aren’t available). Retry after 30 seconds (300 for captions).'],
            ['rate_limited', 'During the session: too many text messages (more than 50 a second for a while, a burst of 100, or more than 5 config or room_test messages a second). The socket closes with 1008.'],
            ['bad_frame', 'During the session: an audio frame we couldn’t read (wrong header, size or codec). The session ends.'],
            ['too_fast', 'During the session: audio arrived faster than real time (bursts up to 6 seconds are fine). The session ends.'],
            ['worker_lost', 'During the session: our processing stopped unexpectedly. The session ends; start again.'],
          ]}
        />
      </Section>

      <Section id="frames" title="2. Audio frames">
        <p>Audio goes both ways as binary messages with a 16 byte header:</p>
        <Code label="Frame layout">{`offset size  field
0      1     version = 1
1      1     codec: 1 = pcm_s16le, 2 = f32le, 3 = opus, 4 = silence
2      2     flags: bit 0 = echo reference (section 5), other bits 0
4      4     seq: frame counter, u32 little endian
8      8     ts_us: your timestamp, u64 (opaque, we send it back)
16     ...   mono 48 kHz samples, one Opus packet (codec 3), or a u16 sample count (codec 4)`}</Code>
        <ul>
          <li><b>Use Opus.</b> Send one 20 ms Opus packet (48 kHz mono, any bitrate; 48 to 64 kbps is plenty) in each frame. We answer with one Opus packet per frame at <code>opusBitrate</code>. That is about 60 kbps each way instead of about 770 kbps of PCM. Opus adds around 6.5 ms each way.</li>
          <li>A frame holds 10 to 100 ms of audio in 10 ms steps (a multiple of 480 samples). 20 ms (960 samples) is recommended.</li>
          <li>We return exactly one frame for each frame you send, with the same <code>seq</code> and <code>ts_us</code>, the same number of samples and the same codec. The cleaned audio is delayed by <code>modelLatencyMs</code> (39 ms), so your total delay is the network round trip plus that. Measure the round trip with <code>ts_us</code>.</li>
          <li>With denoise off (captions only) the audio comes back unchanged and you can ignore it, or start with <code>"returnAudio": false</code> and no audio comes back at all (events still do).</li>
          <li><b>Silence is free.</b> When <code>ready</code> has <code>"silenceFrames": true</code>, send a muted or digitally silent stretch (every sample below 1e-5) as a silence frame instead: the header plus a u16 little-endian sample count (a multiple of 480, at most 4800). We answer it right away with the same silence frame. Silence frames keep the session alive but aren’t billed and cost no processing. Echo reference frames can be silence frames too.</li>
          <li><b>Captions only? You pay for speech, not quiet.</b> With <code>denoise: null</code>, the quiet stretches the Standard captions skip (from 2 s after the last words until someone talks again) and the silence Extra accurate captions don’t send on aren’t billed.</li>
          <li>Send audio in real time (bursts up to 6 seconds are fine). Faster input ends the session with <code>too_fast</code>.</li>
        </ul>
      </Section>

      <Section id="events" title="3. Events from us">
        <p>Text messages, one JSON object each:</p>
        <Code label="Events">{`{"type": "caption", "text": "...", "final": true, "start": 12.34, "end": 15.02}
{"type": "caption", "text": "partial words", "final": false, "id": "item_..."}   // extra accurate only
{"type": "warning", "code": "low_balance", "secondsLeft": 180, "availableMicros": 0}
{"type": "warning", "code": "key_spending_cap", "secondsLeft": 180, "message": "..."}
{"type": "warning", "code": "trial_low", "secondsLeft": 290}
{"type": "warning", "code": "trial_used_up_now_paid", "usdPerMinute": 0.01}
{"type": "warning", "code": "captions_unavailable", "message": "..."}
{"type": "stopped", "reason": "insufficient_funds" | "key_spending_cap" | "max_duration" | "idle_timeout" | "no_audio" | "captions_unavailable" | "trial_over" | "account_closed"}
{"type": "error", "code": "...", "message": "..."}
{"type": "pong", "t": 123, "seconds": 123.4}`}</Code>
        <p><code>start</code> and <code>end</code> are seconds of audio since the session started. Standard captions arrive once for each sentence, 1 to 3 seconds after the speaker pauses; extra accurate captions also stream partial text.</p>
        <p><b>Warnings.</b> <code>low_balance</code>: the balance can&rsquo;t cover more minutes, so the session will stop soon. <code>key_spending_cap</code>: the same, because the key&rsquo;s monthly spending limit leaves no room. <code>trial_low</code>: the free time is almost used up (sent once, at 5 minutes left). <code>trial_used_up_now_paid</code>: the free time ran out and the session went on from your credit, with no gap in the audio. <code>captions_unavailable</code>: extra accurate captions broke down and are not charged from here on.</p>
        <p><b><code>stopped</code></b> is the last thing we send before closing the socket:</p>
        <Table
          head={['Reason', 'Why']}
          rows={[
            ['insufficient_funds', 'The balance can’t cover the next minute.'],
            ['key_spending_cap', 'The key’s monthly spending limit can’t cover the next minute.'],
            ['max_duration', 'A session lasts at most 12 hours.'],
            ['idle_timeout', 'No microphone audio for 2 minutes. Pings and echo reference frames don’t count. A muted microphone must keep sending frames; silent ones are fine.'],
            ['no_audio', 'Microphone audio arrived for less than half of the real time over the last 60 seconds (judged after the first 30). Send every 10 ms of audio in real time, silence included; being a few seconds late is fine, sending a frame now and then is not.'],
            ['captions_unavailable', 'The session was only extra accurate captions and they stopped working. Nothing is charged for them from that point.'],
            ['trial_over', 'The free time ran out and the credit can’t cover the session. Nothing is charged.'],
            ['account_closed', 'The account was suspended or deleted, the key was revoked, or the browser sign-in ended (checked every 30 seconds).'],
          ]}
        />
        <p>A session replaced by a newer one of the same account (see <code>too_many_sessions</code>) is closed with WebSocket code 1001 and no event.</p>
      </Section>

      <Section id="translations" title="Translations">
        <p>Put <code>"translate": ["es", "de"]</code> in <code>start</code>: up to 3 languages as two letter codes, not the spoken one, and it needs <code>captions</code>. <code>ready</code> repeats the list. After every <b>final</b> caption you get one event for each language, usually 1 to 2 seconds later:</p>
        <Code label="Translation event">{`{"type": "translation", "lang": "es", "text": "...", "of": "l12", "start": 12.34, "end": 15.02}`}</Code>
        <p><code>of</code> is the caption&rsquo;s <code>id</code>; <code>start</code> and <code>end</code> are copied from it. Partial captions are never translated. A line that fails is skipped; after 3 failures in a row that language stops for the session, with <code>{'{"type":"warning","code":"translation_unavailable","lang":"de","message":"..."}'}</code>.</p>
        <p>Translations are charged per second of speech translated, for each language, when the session ends (not set aside up front). Sessions with translations are not covered by the free time.</p>
      </Section>

      <Section id="control" title="4. Messages to us">
        <Code label="Control messages">{`{"type": "ping", "t": 123}
{"type": "config", "attenLimDb": 30}
{"type": "config", "id": "c7", "captions": "premium", "language": ["en", "es"], "translate": ["de"]}
{"type": "room_test", "seconds": 20}
{"type": "stop"}`}</Code>
        <p><b>Change anything while live.</b> <code>config</code> takes the same settings as <code>start</code> (<code>denoise</code>, <code>captions</code>, <code>language</code>, <code>translate</code>, <code>echoReference</code>, <code>attenLimDb</code>); what you leave out stays. The audio keeps flowing with the same delay, and you pay for each stretch at the price of the settings it had. Add an <code>id</code> and we answer with the settings now (<code>{'{"type":"config","id":"c7","ok":true,"usdPerMinute":0.0255,...}'}</code>), or with <code>ok: false</code>, a <code>code</code> and a <code>message</code> if it can&rsquo;t be done (live is full, the balance can&rsquo;t cover it, a setting we can&rsquo;t use); then nothing changes. A free-time session that turns on something the free time doesn&rsquo;t cover carries on from your credit.</p>
        <p><b>Measure the room.</b> <code>room_test</code> measures the background noise over the next <code>seconds</code> (5 to 60, 20 if you leave it out) of audio and answers once: <code>{'{"type":"room_test","ok":true,"noiseDb":17.4,"seconds":20}'}</code>. <code>noiseDb</code> is how far the voice stands above the noise that noise removal takes out, the same figure our own recommendations use, whatever strength you picked. Under about 15 the room is very noisy, from about 24 it is quiet. It needs noise removal on; if it is off, or you switch it off before the time is up, you get <code>ok: false</code> with <code>code: "needs_noise_removal"</code> and a <code>message</code>. It counts toward the same limit as <code>config</code>.</p>
        <p> On <code>stop</code> we send the captions that are still waiting, then close. We close the socket after 30 seconds with no message at all (see <code>idle_timeout</code> for what keeps a session alive beyond that). You pay for processed audio seconds only, rounded up to the next second when the session ends.</p>
      </Section>

      <Section id="echo" title="5. Echo cancellation (optional)">
        <p>When someone plays sound through speakers, the microphone hears it again. If your client knows what the speakers play (the &ldquo;reference&rdquo;, for example the desktop sound in OBS), send it along and we take that sound out of the microphone before noise removal and before the captions listen. It is included in the price.</p>
        <ul>
          <li>Put <code>"echoReference": true</code> in <code>start</code>. Only send reference frames when <code>ready</code> says <code>"echoReference": true</code>; an older service ignores the field and would treat the frames as microphone audio.</li>
          <li>A reference frame is an ordinary audio frame with flag bit 0 set, in the same codec as the microphone frames (an Opus client uses a second Opus encoder for the reference). It carries the same <code>seq</code> as the microphone frame it belongs to and is sent before it. Both cover the same 20 ms of the same 48 kHz timeline; a constant offset of tens of milliseconds is fine, we find the delay ourselves (the reference must not come later than the echo it causes).</li>
          <li>We answer microphone frames only. Reference frames get no reply and are neither counted as processed audio nor charged. They are paced like microphone frames.</li>
          <li>A missing or late reference frame counts as silence, so you may skip frames that are all zeros. We keep at most 1 second of reference. The cancellation adapts within a few seconds of the first reference audio.</li>
          <li>In a browser use the browser&rsquo;s own echo cancellation (<code>getUserMedia</code> with <code>echoCancellation: true</code>) instead; no reference is needed.</li>
        </ul>
      </Section>

      <Section id="client" title="Building a client">
        <ul>
          <li>Never block the audio thread: ring buffers in and out, with the network on its own thread.</li>
          <li>Pick a fixed output delay D when you start (round trip plus 30 ms plus a safety margin; the plugin offers 80, 150 and 250 ms) and don&rsquo;t change it during the session. Date the output audio D later. If a cleaned frame misses its time, play the delayed original for that frame (cross-fade by 10 ms or less).</li>
          <li>Reconnect with growing waits and a little randomness. Keep passing the microphone through unprocessed while disconnected, and say why in your status.</li>
          <li>Show the state: connected, round trip, D, late or dropped frames, balance and spending limit warnings.</li>
          <li>Echo cancellation: capture the chosen source, mix it to mono, resample to 48 kHz, keep it in a lock-free buffer, and cut one reference frame for each microphone frame. Skip silent frames. Allow 9 ms more for D.</li>
          <li>Captions: send final captions to your caption output in short lines (about 32 characters or fewer), without queueing a backlog. Other languages (<code>translation</code> events) go to text sources or a subtitle file for each language.</li>
          <li>A stopped session isn&rsquo;t the end. After <code>no_audio</code> or <code>idle_timeout</code>, reconnect as soon as audio flows again, not after a long fixed wait. Whatever ended a session, keep the microphone passing through.</li>
        </ul>
      </Section>
    </DevPage>
  );
}
