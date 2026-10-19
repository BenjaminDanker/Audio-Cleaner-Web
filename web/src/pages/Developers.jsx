import { Link } from 'react-router';
import { Code, DevPage, Section, Table, apiBase } from '../components/DevDocs.jsx';
import { usePageTitle } from '../hooks.js';

// Written by hand from docs/api.md (and docs/streaming-protocol.md for the live page): when the API changes, change all of them.
// Plain words, no internals (the copy test tests/copy.mjs checks this page; text inside code samples is exempt there
// because the API's own field values, like "premium", have to be spelled exactly).

const TOC = [
  ['start', 'Quick start'],
  ['keys', 'Keys and scopes'],
  ['files', 'Cleaning a file'],
  ['options', 'All the options'],
  ['prices', 'Prices and estimates'],
  ['safe', 'Safe retries'],
  ['again', 'Changing a finished job'],
  ['callbacks', 'Finish notifications'],
  ['limits', 'Spending limits'],
  ['billing', 'What a call costs'],
  ['errors', 'Errors'],
  ['rates', 'Rate limits'],
  ['live', 'Live streaming'],
];

export default function Developers() {
  usePageTitle('For developers');
  const base = apiBase();
  const hdr = '-H "Authorization: Bearer $KEY"';
  return (
    <DevPage
      title="For developers"
      lead="Send files to Audio Cleaner from your own scripts, and use the same balance and prices as the website. Everything here is plain HTTP and JSON."
      toc={TOC}
    >
      <Section id="start" title="Quick start">
        <ol className="ol-steps">
          <li>Sign in, open <Link to="/app/live">Live &amp; keys</Link> and create a key with <b>File API for scripts</b> switched on. Copy it: it is shown only once.</li>
          <li>Make sure your balance can cover the job (<Link to="/app/billing">Billing</Link>).</li>
          <li>Create a job, upload the file, wait for it to finish, download the results.</li>
        </ol>
        <p>Every address on this page is relative to:</p>
        <Code label="Base address">{base}</Code>
        <Code label="Clean one file">{`BASE=${base}
KEY=acw_...        # your key

# 1. tell us about the file (nothing is charged yet, the price is only set aside)
curl -s $BASE/api/jobs ${hdr} -H "Content-Type: application/json" \\
  -d '{"kind":"denoise","denoiseTier":"standard","filename":"talk.wav","sizeBytes":'$(stat -c %s talk.wav)',"durationSeconds":612.4}'
# -> {"id":"3f2c...","estCostMicros":102167,"uploadUrl":"api/jobs/3f2c.../input"}

# 2. send the file itself, as the raw body
curl -s -X PUT --data-binary @talk.wav ${hdr} $BASE/api/jobs/3f2c.../input

# 3. check on it every few seconds until "status" is "done"
curl -s ${hdr} $BASE/api/jobs/3f2c...

# 4. download the result (use a name from "outputs")
curl -s -OJ ${hdr} $BASE/api/jobs/3f2c.../outputs/talk-cleaned.wav`}</Code>
        <p className="small muted">Money is always in <b>micros</b>: 1,000,000 micros = 1 US dollar. So <code>estCostMicros: 102167</code> is ten cents and a bit.</p>
      </Section>

      <Section id="keys" title="Keys and scopes">
        <p>A key lets a program act on your account and spend your balance, so keep it like a password. You can have up to 10 active keys, create and revoke them on the <Link to="/app/live">Live &amp; keys</Link> page, and give each one only what it needs:</p>
        <Table
          head={['Scope', 'What it allows']}
          rows={[
            ['jobs', 'The file API on this page: create, estimate, upload, list, download, cancel and delete jobs.'],
            ['stream', 'Live audio over a WebSocket (the OBS plugin uses this). See the live page.'],
          ]}
        />
        <p>A key can never open billing, change keys or touch your account details; those need you signed in on the website. A request outside the key&rsquo;s scopes gets <code>403</code>. Besides the file and live APIs, a key can read <code>GET /api/me</code> (your balance, your free live time left and the key&rsquo;s own spending limit) and the public price list.</p>
        <p>Send the key in the <code>Authorization</code> header of every request:</p>
        <Code label="Authorization header">{`Authorization: Bearer acw_...`}</Code>
        <p>A key that is wrong or revoked gets <code>401 {'{"detail":"Invalid API key."}'}</code>. Using the API means you accept the current <Link to="/terms">Terms</Link>.</p>
      </Section>

      <Section id="files" title="Cleaning a file">
        <p>A file goes through five steps. Only the first two touch your balance, and only to set money aside; you pay when the job is done.</p>
        <h3>1. Create the job</h3>
        <p><code>POST /api/jobs</code> with JSON. The answer has the job&rsquo;s <code>id</code>, what is set aside (<code>estCostMicros</code>) and where to send the file (<code>uploadUrl</code>, relative to the base address and without a leading slash).</p>
        <Code label="Create a job">{`curl -s $BASE/api/jobs ${hdr} -H "Content-Type: application/json" -d '{
  "kind": "denoise_transcribe",
  "denoiseTier": "standard",
  "transcribeTier": "local",
  "filename": "interview.mp4",
  "sizeBytes": 48211003,
  "durationSeconds": 1810.5,
  "displayName": "Interview with Sam",
  "options": {"audio_format": "flac", "language": "en", "translate": ["es"]}
}'`}</Code>
        <Table
          head={['Field', 'What it is']}
          rows={[
            ['kind', '"denoise" cleans the sound, "transcribe" writes subtitles and a transcript, "denoise_transcribe" does both.'],
            ['denoiseTier', 'For cleaning: "standard" (Denoising), "rebuild" (Voice rebuild: the whole voice is rebuilt, for room echo, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud; own price; files up to 90 minutes), or "off" (no noise removal, only Leveling and/or Restore highs: at least one of them must be on). Leveling (options.leveling, on by default) and Restore highs (options.restore_highs, off by default) are each their own price.'],
            ['transcribeTier', <>For subtitles: <code>"local"</code> (Standard) or <code>"premium"</code> (extra accurate; the audio goes to a speech partner, see the <Link to="/privacy">Privacy Policy</Link>).</>],
            ['filename', 'Its ending must be one we can read. Audio: wav, rf64, w64, mp3, mp2, mpa, m4a, m4b, m4r, aac, flac, ogg, oga, opus, spx, wma, aiff, aif, aifc, caf, au, snd, amr, ac3, eac3, dts, mka, wv, ape, tta, mpc. Video: mp4, m4v, mov, webm, mkv, avi, wmv, asf, flv, f4v, 3gp, 3g2, ts, mts, m2ts, mpg, mpeg, vob, mxf, dv, ogv, rm, rmvb. Anything else is refused with "Unsupported file type." Only the ending is used; the name never becomes a path.'],
            ['sizeBytes', 'The exact size of the upload: 1 byte up to 4 GiB. A file can weigh at most 40 MB per second of its length (small files, under 200 MB, are always fine).'],
            ['durationSeconds', 'The length of the audio: from 0.5 seconds to 4 hours (14,400), or 90 minutes (5,400) when the voice is rebuilt. We measure it again after the upload and price what we find.'],
            ['displayName', 'Optional. A friendlier name for the job and its result files (up to 120 characters).'],
            ['callbackUrl', 'Optional. Where to tell you when the job ends, see Finish notifications.'],
            ['options', 'The choices below. Keep the whole object under 2,000 characters.'],
          ]}
        />
        <p>An account can have 3 file jobs in progress at once (waiting or running, and changing a finished job counts too) and up to 8 GiB of uploads waiting. The next one is refused with <code>429</code> and the code <code>too_many_active_jobs</code>: &ldquo;You have 3 files being processed. Start this one when one of them is done.&rdquo; Try again when one has finished. If your balance can&rsquo;t cover the estimate you get <code>402</code> and nothing is created.</p>

        <h3>2. Upload the file</h3>
        <p><code>PUT</code> the raw bytes to the <code>uploadUrl</code>: no JSON, no form. The <code>Content-Length</code> header is required and must equal <code>sizeBytes</code> (<code>411</code> if not). Start within 10 minutes of creating the job. The answer is <code>{'{"ok":true}'}</code>.</p>
        <p>If the upload fails, make a new job. Errors here: <code>409</code> when the job isn&rsquo;t waiting for a file (already uploaded or canceled), <code>408</code> after 30 seconds without data, <code>413</code> when more arrives than you said, <code>400</code> when less does, <code>507</code> when we are short on storage for a moment.</p>

        <h3>3. Wait for it</h3>
        <p><code>GET /api/jobs/&#123;id&#125;</code> every few seconds (every 2 seconds is fine), or let us tell you with a <a href="#callbacks">finish notification</a>. To watch several jobs, <code>GET /api/jobs</code> lists up to 50 of your jobs in one request: the ones in progress and the ones whose files are still there (2 hours after they finished).</p>
        <Code label="A finished job">{`{
  "id": "3f2c...", "kind": "denoise", "label": "Denoising + Leveling",
  "status": "done", "stage": "Done", "progress": 100, "error": null,
  "inputName": "Interview with Sam", "durationSeconds": 1810.5,
  "estCostMicros": 301834, "costMicros": 301834,
  "notice": null, "queuePosition": null, "queueWaitSeconds": null,
  "options": {"audio_format": "flac", "...": "..."},
  "outputs": [{"name": "interview-cleaned.flac", "kind": "audio", "bytes": 31457280}],
  "createdAt": 1760600000, "finishedAt": 1760600420, "expiresAt": 1760686820
}`}</Code>
        <ul>
          <li><b>status</b> goes <code>uploading</code>, <code>queued</code>, <code>running</code>, then <code>done</code>, <code>failed</code> or <code>canceled</code> (and <code>expired</code> once the results are deleted). If our service restarts, your job simply goes back in line and starts again.</li>
          <li><b>stage</b> and <b>progress</b> (0 to 100) say what it is doing. While <code>queued</code>, <code>queuePosition</code> and <code>queueWaitSeconds</code> (an estimate) tell you how long.</li>
          <li><b>estCostMicros</b> is what was set aside; <b>costMicros</b> is what you were charged (0 if it failed or was canceled).</li>
          <li><b>notice</b> is a friendly note about how your file was handled (for example &ldquo;the file had only one audio track&rdquo;). It is not an error. <b>error</b> says why a job failed or was canceled.</li>
          <li><b>outputs</b> lists the result files. <code>kind</code> is <code>audio</code>, <code>video</code>, <code>captions</code>, <code>transcript</code> or <code>segments</code>; captions and transcripts in other languages have a <code>lang</code>.</li>
        </ul>

        <h3>4. Download the results</h3>
        <p><code>GET /api/jobs/&#123;id&#125;/outputs/&#123;name&#125;</code> with a <code>name</code> from <code>outputs</code>. The file comes as an attachment. Your upload, the copies kept for <a href="#again">changing the job</a> and the results are kept together for <b>2 hours</b> after the job is done, then deleted together for good.</p>
        <p>A video comes back in its own container when it can hold the result (mp4, mov, m4v, mkv, webm), otherwise as MKV. Other audio tracks are kept as they were; only the first audio track is cleaned. Subtitles go into the video where the container can carry them.</p>

        <h3>5. Cancel or delete</h3>
        <Code label="Cancel and delete">{`curl -s -X POST   ${hdr} $BASE/api/jobs/3f2c.../cancel
curl -s -X DELETE ${hdr} $BASE/api/jobs/3f2c...`}</Code>
        <p><code>cancel</code> works while the job is still <code>uploading</code> or <code>queued</code>, and gives everything set aside back. Once it has started, or once the extra accurate transcription is done and only the packaging is left, you get <code>409</code>. <code>DELETE</code> removes everything of the job right now: the upload, the kept copies and the results, and the same for every job made from it by changing it (<code>409</code> while the job, or one made from it, is still in progress).</p>
      </Section>

      <Section id="options" title="All the options">
        <p>Everything goes in the <code>options</code> object. Anything you leave out gets its default. A value that is out of range is refused with <code>400</code> and a message saying what to change; nothing is silently guessed, except <code>atten_lim_db</code>, which is kept between 6 and 100.</p>
        <Table
          head={['Option', 'Default', 'Values and meaning']}
          rows={[
            ['audio_format', 'flac', 'The format of the cleaned audio: wav, flac, mp3, m4a or opus.'],
            ['caption_formats', '["srt","vtt","txt"]', 'Any of srt, vtt, txt and json. json gives a "segments" file with every line and its times.'],
            ['language', 'null', 'The language spoken, as a 2 or 3 letter code ("en", "es", "haw"). null means we find it ourselves.'],
            ['translate', '[]', 'Up to 3 languages to translate the subtitles and transcript into (two letter codes: en, es, fr, de, it, pt, nl, pl, ru, uk, tr, ar, hi, ja, ko, zh, sv, da, fi, no, cs, el, he, id, vi), not the language spoken. Needs "transcribe" or "denoise_transcribe". Each language costs extra; a language that can’t be made is left out, said in "notice" and not charged.'],
            ['output_video', 'true', 'For a video: give a video back with the cleaned sound. false gives only the audio.'],
            ['embed_subtitles', 'true', 'Put the subtitles inside the video. This needs the whole video uploaded.'],
            ['loudness_lufs', '-16', 'The loudness to aim for, a number from -30 to -9 (podcast -16, online video -14, broadcast -23), or null to leave the level alone. Outside the range is a 400.'],
            ['atten_lim_db', '100', 'How much noise may be taken out, in dB, kept between 6 (gentle) and 100 (as much as possible). The website offers 12, 24, 40 and 100. Does not apply when the voice is rebuilt.'],
            ['noise', 'auto', '"auto" measures the recording and picks; "light" suits hiss, hum and a quiet room; "heavy" suits noise almost as loud as the voice. Does not apply when the voice is rebuilt.'],
            ['echo_reference_track', 'false', 'true or false (anything else is a 400). Removes speaker echo from the first audio track by using the second one (the sound from your speakers, as OBS can record it). Included in the price.'],
            ['leveling', 'true', 'true or false. Leveling, with its own price: evens out the voice and sets the loudness. false returns the cleaned voice as it is.'],
            ['restore_highs', 'false', 'true or false. Restore highs, with its own price and its own line on the bill (per second, like Leveling). When it is true it always runs: it brings back the high sounds of a recording that lost them (phone calls, voice messages, small files), and a recording that already has them gets the widest restoration. Works with Denoising, or on its own with denoiseTier "off". Not with the voice rebuilt, which makes its own highs; it is not charged then.'],
          ]}
        />
        <p>Restore highs used to be on by default, free, and only did something for recordings that had lost their highs. It is now off unless you set <code>restore_highs</code> to true, is charged like Leveling, and does its work whenever it is on; the job&rsquo;s <code>notice</code> says when it happened. Jobs made before this change keep the prices they were made with.</p>
      </Section>

      <Section id="prices" title="Prices and estimates">
        <p>The prices are per minute of audio, counted by the second and rounded up to the next second. <code>GET /api/billing/pricing</code> is public (no key needed) and lists them, plus the limits:</p>
        <Code label="Price list">{`curl -s $BASE/api/billing/pricing
{
  "rates": [
    {"product": "denoise", "tier": "standard", "label": "Denoising", "usdPerMinute": 0.009, "usdPerHour": 0.54, "available": true},
    {"product": "leveling", "tier": "standard", "label": "Leveling", "usdPerMinute": 0.001, "usdPerHour": 0.06, "available": true},
    {"product": "transcribe_translate", "tier": "standard", "label": "Translated subtitles, per language", "usdPerMinute": 0.004, "...": "..."}
  ],
  "maxJobSeconds": 14400, "maxRebuildSeconds": 5400, "maxUploadBytes": 4294967296,
  "topup": {"minCents": 500, "maxCents": 50000, "presetsCents": [1000, 2000, 5000, 10000]},
  "paymentsEnabled": true, "taxAddedAtCheckout": false
}`}</Code>
        <p>The products are <code>denoise</code>, <code>transcribe</code>, <code>transcribe_translate</code> (one translation, per language), and for live <code>stream_denoise</code>, <code>stream_captions</code> and <code>stream_translate</code>. The prices on the <Link to={{ pathname: '/', hash: '#pricing' }}>home page</Link> come from the same list.</p>
        <p>To know what a job will cost before you create it, ask <code>POST /api/jobs/estimate</code>. It takes the same <code>kind</code>, <code>denoiseTier</code>, <code>transcribeTier</code>, <code>durationSeconds</code> and <code>options</code> as creating a job (no file fields), checks them the same way, and creates nothing and sets nothing aside.</p>
        <Code label="Estimate a job">{`curl -s $BASE/api/jobs/estimate ${hdr} -H "Content-Type: application/json" -d '{
  "kind": "denoise_transcribe", "denoiseTier": "standard", "transcribeTier": "local",
  "durationSeconds": 1810.5, "options": {"translate": ["es", "de"]}
}'
{
  "estCostMicros": 664036, "estCostUsd": 0.664036, "durationSeconds": 1810.5, "maxSeconds": 14400,
  "lines": [
    {"item": "denoise",    "label": "Denoising",                     "quantity": 1, "usdPerMinute": 0.009, "micros": 271650},
    {"item": "leveling",   "label": "Leveling",                      "quantity": 1, "usdPerMinute": 0.001, "micros": 30184},
    {"item": "transcribe", "label": "Transcription",                      "quantity": 1, "usdPerMinute": 0.004, "micros": 120734},
    {"item": "translate",  "label": "Translated subtitles, per language", "quantity": 2, "usdPerMinute": 0.004, "micros": 241468}
  ],
  "options": {"...": "your options with the defaults filled in"},
  "availableMicros": 5000000, "affordable": true
}`}</Code>
        <p>The total is exactly what creating the same job would set aside. If your key has a spending limit, the answer also has <code>keyLimit</code> with what is left of it.</p>
      </Section>

      <Section id="safe" title="Safe retries">
        <p>If your connection drops right after you create a job, you can&rsquo;t tell whether it was made. Add an <code>Idempotency-Key</code> header (1 to 128 visible characters) and ask again with the same one: the same job comes back instead of a second job and a second reservation.</p>
        <Code label="Idempotency key">{`curl -s $BASE/api/jobs ${hdr} -H "Content-Type: application/json" \\
  -H "Idempotency-Key: interview-2026-10-16" -d '{ ... }'
# the same call again, within 24 hours:
# -> the same {"id":..., "estCostMicros":..., "uploadUrl":..., "idempotentReplay": true}
#    with the header  Idempotent-Replayed: true`}</Code>
        <ul>
          <li>Keys are remembered for 24 hours, per account. Use a new key for each new file.</li>
          <li>The details must be the same. The same key with a different body is <code>409</code>, so a typo can&rsquo;t quietly start something else.</li>
          <li>A request that failed (for example <code>402</code> because the balance was too low) doesn&rsquo;t use the key up; try again with it once you have fixed the cause.</li>
          <li>A repeat doesn&rsquo;t count against the hourly limit for creating jobs.</li>
        </ul>
      </Section>

      <Section id="again" title="Changing a finished job">
        <p>For 2 hours after a job is done, the files it needs to be changed are kept next to its results: your upload, the noise removal before its strength is applied, the rebuilt voice when there was one, and the transcript. In that time you can ask for the same file again with other choices. Only the steps whose inputs changed run again, and you pay only for the steps that cost something and really run again. Setting the strength, restoring highs and writing the file cost nothing.</p>
        <Table
          head={['You change', 'What runs again']}
          rows={[
            ['atten_lim_db', 'Nothing but a mix of the kept noise removal and your recording, which is free. Leveling comes after it, so with Leveling on (the default) it runs again and is charged; send leveling: false as well to change only the strength for free.'],
            ['restore_highs, leveling, loudness_lufs', 'From the kept noise removal: Restore highs and Leveling are charged at their prices whenever they run (restore_highs: true always runs); with both off it is free.'],
            ['denoiseTier: "rebuild"', 'The voice is rebuilt from your upload, at the full Voice rebuild price.'],
            ['denoiseTier: "standard" (from a rebuild)', 'The kept noise removal is used. If the job never had one, noise removal runs and is charged.'],
            ['noise, echo_reference_track', 'Noise removal runs again from your upload and is charged.'],
            ['transcribeTier, language', 'The transcript is written again from your original sound and is charged. Other languages in translate cost only for the new ones.'],
            ['audio_format, caption_formats', 'The files are written again from what is already there. Free.'],
          ]}
        />
        <h3>Ask for the price, then do it</h3>
        <p><code>POST /api/jobs/&#123;id&#125;/rerun/estimate</code> takes the choices you want to change (the rest stay as they were): <code>denoiseTier</code>, <code>transcribeTier</code> (<code>"off"</code> leaves the subtitles out; on its own that changes nothing) and <code>options</code>. It runs nothing and sets nothing aside, and answers with what would run, its price as lines, and whether your balance can cover it. <code>POST /api/jobs/&#123;id&#125;/rerun</code> with the same body does it: it sets the price aside, makes a new job with its own <code>id</code> (watch it, download from it and delete it like any other), and answers <code>{'{"id", "estCostMicros", "free", "expiresAt"}'}</code>. Add <code>maxCostMicros</code> with the price you were quoted and the call is refused with <code>409</code> and the code <code>price_changed</code> if it is higher now.</p>
        <Code label="Change a finished job">{`# what would it cost to keep Leveling on at a gentler strength?
curl -s $BASE/api/jobs/3f2c.../rerun/estimate ${hdr} -H "Content-Type: application/json" \
  -d '{"options": {"atten_lim_db": 24}}'
# -> {"estCostMicros": 30184, "free": false, "lines": [{"item": "leveling", "label": "Leveling", "micros": 30184, ...}],
#     "steps": {"clean": "denoised", "transcribe": false, "translate": []}, "expiresAt": 1760686820, "affordable": true, ...}

# do it
curl -s $BASE/api/jobs/3f2c.../rerun ${hdr} -H "Content-Type: application/json" \
  -d '{"options": {"atten_lim_db": 24}, "maxCostMicros": 30184}'
# -> {"id": "9a77...", "estCostMicros": 30184, "free": false, "expiresAt": 1760686820}`}</Code>
        <ul>
          <li>The new job goes through the same line and the same limits as any job: it counts toward the jobs in progress, and a key&rsquo;s spending limit holds for it.</li>
          <li>It never extends the time the files are kept: the new job&rsquo;s results go at the same moment as the job it came from, and so does your upload. The answer&rsquo;s <code>expiresAt</code> says when. A change that is still only finishing in the last minute of that time is stopped and not charged.</li>
          <li>A key can only change jobs that key made. Your signed-in website session can change any job of the account, including ones a key made. Anything else is <code>404</code> with the code <code>job_not_found</code>, the same answer as for a job that doesn&rsquo;t exist.</li>
          <li>In your credit history a paid change is one line, &ldquo;Changed and saved again&rdquo; and the steps it did. A free one has no line.</li>
          <li>A voice rebuild needs the same things as creating one (files up to 90 minutes, and it can be paused for a moment).</li>
        </ul>
        <Table
          head={['Status and code', 'What it means']}
          rows={[
            ['404 job_not_found', 'No such job, or not one this key made.'],
            ['409 job_files_gone', 'The job’s files are already deleted (2 hours passed, or it was deleted): start again from your file.'],
            ['409 job_not_finished', 'Only a finished job can be changed.'],
            ['409 job_expiring', 'Too little of the 2 hours is left for the work to finish. Start again from your file.'],
            ['409 price_changed', 'The price is above your maxCostMicros.'],
            ['400 nothing_to_change', 'These choices give the files the job already has.'],
            ['429 too_many_active_jobs', 'You already have the most file jobs in progress that an account may have. Try again when one is done.'],
            ['402, 403', 'As for creating a job: not enough balance, or the key’s monthly spending limit (X-Error-Code: key_spending_cap).'],
          ]}
        />
      </Section>

      <Section id="callbacks" title="Finish notifications">
        <p>Instead of checking again and again, give a <code>callbackUrl</code> when you create the job. When the job is done, fails or is canceled we send one small JSON message to that address.</p>
        <Code label="What we send">{`POST /hooks/audio-cleaner HTTP/1.1
Content-Type: application/json
X-Audio-Cleaner-Signature: t=1760600420,v1=5b1c0a...

{"type": "job.done", "jobId": "3f2c...", "status": "done", "error": null, "costMicros": 301834,
 "outputs": [{"name": "interview-cleaned.flac", "kind": "audio", "bytes": 31457280}],
 "finishedAt": 1760600420, "expiresAt": 1760686820}`}</Code>
        <p><code>type</code> is <code>job.done</code>, <code>job.failed</code> or <code>job.canceled</code>. The message is small on purpose: use <code>GET /api/jobs/&#123;id&#125;</code> and the download address for the rest. Answer with any <code>2xx</code> status to say you got it. A message can arrive twice (we try again when an answer is lost), so treat <code>jobId</code> as the thing to de-duplicate on.</p>
        <h3>Check that it is really us</h3>
        <p>Every message is signed with your account&rsquo;s secret (it starts with <code>whsec_</code>). You see it, and can replace it, in the <Link to="/app/live">API keys card</Link>; it is made the first time you use a notification. The header holds a time and a signature: <code>v1</code> is the HMAC-SHA256 of the text <code>&lt;t&gt;.&lt;the raw body&gt;</code> with your secret, in hex. Compare in constant time, and refuse a time that is more than 5 minutes old.</p>
        <Code label="Check the signature (Python)">{`import hashlib, hmac, time

def is_from_audio_cleaner(secret: str, header: str, raw_body: bytes) -> bool:
    parts = dict(p.split("=", 1) for p in header.split(","))
    if abs(time.time() - int(parts["t"])) > 300:
        return False
    wanted = hmac.new(secret.encode(), parts["t"].encode() + b"." + raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(wanted, parts["v1"])`}</Code>
        <Code label="Check the signature (Node)">{`import { createHmac, timingSafeEqual } from 'node:crypto';

export function isFromAudioCleaner(secret, header, rawBody) {
  const p = Object.fromEntries(header.split(',').map((x) => x.split('=')));
  if (Math.abs(Date.now() / 1000 - Number(p.t)) > 300) return false;
  const wanted = createHmac('sha256', secret).update(\`\${p.t}.\${rawBody}\`).digest('hex');
  return wanted.length === p.v1.length && timingSafeEqual(Buffer.from(wanted), Buffer.from(p.v1));
}`}</Code>
        <h3>Where we will send it</h3>
        <ul>
          <li>The address must start with <code>https://</code>, use the normal port 443, and have a public host name: no IP address, no <code>user:password@</code>, no <code>#</code> part, at most 500 characters. Otherwise creating the job fails with <code>400</code>.</li>
          <li>For your safety and ours we never call addresses inside a private network: a name that points to a private, local or reserved address (IPv4 or IPv6) is refused when we send. We don&rsquo;t follow redirects (a <code>301</code> counts as a failure), wait at most 5 seconds for an answer, and try up to 3 times: right away, about 30 seconds later, and about 5 minutes after that.</li>
          <li>Your certificate has to be valid for the name.</li>
          <li>What happened shows with the job, as <code>callback</code>: <code>{'{"url", "state", "attempts", "lastStatus", "lastError", "deliveredAt", "nextAttemptAt"}'}</code>. <code>state</code> is <code>pending</code>, <code>delivered</code> or <code>failed</code>, and <code>lastError</code> says in words what went wrong. The query part of your address is not shown back. A notification that never gets through is not tried again later; check the job yourself.</li>
        </ul>
      </Section>

      <Section id="limits" title="Spending limits">
        <p>A key can carry a limit for each calendar month (UTC), so a script that goes wrong, or a key that leaks, can only spend so much. Set it when you create the key, or change or remove it later in the <Link to="/app/live">API keys card</Link>, which also shows what each key has used this month.</p>
        <ul>
          <li>What counts: finished jobs and live sessions at their real price, and running ones at what is set aside for them. Your own sign-in on the website and your other keys are not affected.</li>
          <li>A job that would go over the limit is refused with <code>403</code> before anything is set aside. The message says how much is left and when the month starts over, and the response has the header <code>X-Error-Code: key_spending_cap</code> so a script can tell it from other 403s.</li>
          <li>Live sessions: starting one that would go over is refused with the error <code>key_spending_cap</code>; a running one gets a warning with that code when the next minute doesn&rsquo;t fit, and stops (<code>stopped</code> with the reason <code>key_spending_cap</code>) when the minutes already set aside run out. The OBS plugin shows the message.</li>
          <li>A job that is already accepted is never stopped by the limit. If a file turns out longer than you said, it can finish a little above it.</li>
          <li>Ask what is left with <code>GET /api/me</code>: it has <code>keyLimit</code> with <code>monthlyCapMicros</code>, <code>usedMicros</code>, <code>remainingMicros</code> and <code>resetsAt</code> for a key that has a limit.</li>
        </ul>
      </Section>

      <Section id="billing" title="What a call costs">
        <p>You only pay for audio that was really processed, and a job that fails costs nothing.</p>
        <ol className="ol-steps">
          <li><b>Set aside.</b> Creating a job sets aside its estimated price (<code>estCostMicros</code>) from your balance, at today&rsquo;s prices. If your balance can&rsquo;t cover it, nothing happens and you get <code>402</code>. The price you saw stays the price for that job, even if the list changes while it waits.</li>
          <li><b>Measure.</b> After the upload we measure the real length. If the file is longer than you said, the amount set aside grows to match; if your balance can&rsquo;t cover it, the job stops and you aren&rsquo;t charged.</li>
          <li><b>Charge.</b> When the job is done, the real price (<code>costMicros</code>) is taken and the rest is given back. A translation that couldn&rsquo;t be made isn&rsquo;t charged.</li>
          <li><b>Give back.</b> A job that fails, is canceled, or never gets its file gives back everything it had set aside.</li>
        </ol>
        <p>Live sessions work the same way: about five minutes are set aside when a session starts and more is added a minute at a time; at the end you pay for the seconds of audio that were processed, translations at the end. The free live time uses no balance at all.</p>
      </Section>

      <Section id="errors" title="Errors">
        <p>An error is always JSON, <code>{'{"detail": "what went wrong, in words"}'}</code>, with one of these statuses. Where a script may want to tell one error from another, the answer also has <code>"code"</code> (and the same in the header <code>X-Error-Code</code>): <code>too_many_active_jobs</code>, <code>key_spending_cap</code> and the codes of <a href="#again">changing a job</a>. A malformed request body is <code>422</code> with <code>{'{"detail":"Invalid request.","fields":[...]}'}</code> naming the fields.</p>
        <Table
          head={['Status', 'What it means']}
          rows={[
            ['400', 'Something in the request isn’t allowed: an option out of range, a file ending we don’t read, a length outside the limits, a size that doesn’t fit the length, a bad callbackUrl or Idempotency-Key. The message says which.'],
            ['401', 'No key, or the key is wrong or revoked.'],
            ['402', 'Your available balance can’t cover the estimated price. The message has both amounts.'],
            ['403', 'The key doesn’t have the scope, the account is suspended, or the key’s monthly spending limit is reached (header X-Error-Code: key_spending_cap).'],
            ['404', 'The job doesn’t exist, isn’t yours, or the file isn’t available (yet, or any more).'],
            ['408', 'An upload sent nothing for 30 seconds.'],
            ['409', 'Not possible in the job’s current state (uploading twice, canceling after it started, deleting while it runs), or an Idempotency-Key was used for different details.'],
            ['411', 'The upload has no Content-Length, or it is not the sizeBytes you gave.'],
            ['413', 'The file is bigger than the 4 GiB limit, or the upload is longer than you said.'],
            ['422', 'The request body isn’t shaped right (see "fields").'],
            ['429', 'Too many requests. Wait for the number of seconds in the Retry-After header. With the code too_many_active_jobs: the account already has 3 file jobs in progress; try again when one is done.'],
            ['503', 'The service is taking a short break. Try again in a minute.'],
            ['507', 'We are short on storage for a moment. Try again a bit later.'],
          ]}
        />
      </Section>

      <Section id="rates" title="Rate limits">
        <p>These keep things fair for everyone. When you hit one the answer is <code>429</code> with a <code>Retry-After</code> header (in seconds) and a message that says how long to wait.</p>
        <Table
          codeFirst={false}
          head={['What', 'Limit']}
          rows={[
            ['Any API call', 'A burst of 120, then 2 a second, for each network. Polling one job every 2 seconds is fine; to watch many, list them with GET /api/jobs.'],
            ['Creating jobs', '20 an hour for each account and each network. A repeat of an Idempotency-Key doesn’t count.'],
            ['Uploads', '30 an hour for each account. Upload speed is shared out fairly.'],
            ['Jobs at once', '3 file jobs in progress for each account (changing a finished job counts), and no more than 8 GiB of uploads waiting.'],
            ['Live sessions', '60 starts an hour for each account and 4 running at once; see the live page.'],
          ]}
        />
        <p>We only ever add fields to answers; we don&rsquo;t rename or remove them. Your code should ignore fields it doesn&rsquo;t know.</p>
      </Section>

      <Section id="live" title="Live streaming">
        <p>Live noise removal and live captions (with optional translations) run over one WebSocket per audio source, <code>wss://&hellip;/api/stream/ws</code>, authorised with a key that has the <code>stream</code> scope. You send 20 millisecond Opus frames of microphone audio and get cleaned frames back with captions as text messages. The OBS plugin does exactly this.</p>
        <ul>
          <li>Everything you need to build your own client is on <Link to="/developers/live">the live protocol page</Link>: the handshake, the start message, audio frames, events, errors and echo cancellation.</li>
          <li>Price per minute comes back in the <code>ready</code> message as <code>usdPerMinute</code>; about five minutes are set aside when you start and the rest is paid by the second.</li>
          <li>New accounts get two hours of Standard live for free, once.</li>
        </ul>
      </Section>
    </DevPage>
  );
}
