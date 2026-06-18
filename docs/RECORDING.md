# Terminal Recording

Use this workflow to capture Avalan CLI sessions as real terminal video,
including Rich colors, live panels, spinners, and tool-call updates.

The important distinction is that `script` captures a transcript, not a
video. For live terminal behavior, record a timed terminal stream first,
then render that stream to an animation or MP4.

## Requirements

Install or verify these tools:

```bash
command -v ttyrec ffmpeg
```

On macOS with a visible Terminal.app window, `ttygif` can convert a
`ttyrec` file to GIF:

```bash
command -v ttygif
```

For headless runs, use a ttyrec-aware renderer or an ANSI-to-frames helper
to turn the `.ttyrec` file into PNG frames before calling `ffmpeg`.

## Capture

Record the command with `ttyrec`. Force terminal color explicitly because
developer shells may set `NO_COLOR=1`, and Rich will otherwise remove ANSI
styles from the capture.

```bash
env -u NO_COLOR \
  TERM=xterm-256color \
  COLORTERM=truecolor \
  FORCE_COLOR=1 \
  TTY_COMPATIBLE=1 \
  COLUMNS=120 \
  LINES=40 \
  ttyrec -e "printf '%s\n' 'What is (4 + 6) and then that result times 5, divided by 2? Use the calculator tool.' \
    | .venv/bin/avalan agent run \
        --engine-uri 'ai://local//Users/mariano/Code/ai/pyds4/.local/ds4/ds4flash.gguf?backend=ds4&ds4_ctx=4096&ds4_native_backend=metal' \
        --tool 'math.calculator' \
        --run-max-new-tokens 512 \
        --run-temperature 0 \
        --reasoning-effort high \
        --role 'You are a helpful assistant named Tool, that can resolve user requests using tools.' \
        --display-events \
        --display-tools \
        --display-tools-events 10 \
        --display-answer-height-expand \
        --stats" \
  /private/tmp/avalan-recording.ttyrec
```

For Avalan agent recordings, choose the diagnostics you want independently:
`--stats` shows token generation statistics, `--display-events` shows
non-tool stream events, and `--display-tools` shows tool lifecycle details
and results. Tool and event diagnostics do not require `--stats`.

Avalan uses one live owner to render the active terminal view for all roles.
When `--record` is enabled, recording saves after the owner render so the
captured frame matches the displayed live frame. Rapid updates may coalesce
to the latest live frame instead of preserving every intermediate visual
state, while lossless canonical/public response surfaces remain intact.

Use `ttyplay` to inspect the capture:

```bash
ttyplay /private/tmp/avalan-recording.ttyrec
```

## Convert With ttygif

If you are running inside a visible macOS Terminal.app window, convert the
recording to GIF:

```bash
REPO=/Users/mariano/Code/ai/avalan
cd /private/tmp
TERM_PROGRAM=Apple_Terminal ttygif /private/tmp/avalan-recording.ttyrec
```

Then convert the GIF to MP4:

```bash
ffmpeg -y \
  -i /private/tmp/tty.gif \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$REPO/docs/avalan-recording.mp4"
```

`ttygif` uses macOS `screencapture`. It may fail or hang in headless
automation sessions that do not have access to a real Terminal.app window.
In that case, keep the `.ttyrec` file and use the headless path below.

## Convert Headlessly

In a headless Codex session, render the `.ttyrec` stream to PNG frames with
a local ANSI renderer, then encode those frames:

```bash
python /path/to/ttyrec_to_frames.py \
  /private/tmp/avalan-recording.ttyrec \
  /private/tmp/avalan-recording-frames \
  --cols 80 \
  --rows 40 \
  --fps 10

ffmpeg -y \
  -framerate 10 \
  -i /private/tmp/avalan-recording-frames/frame_%05d.png \
  -vf "scale=1600:-2:flags=lanczos" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  docs/avalan-recording.mp4
```

Keep the raw `.ttyrec` next to the rendered video while reviewing. It is the
source of truth if the renderer drops a glyph, misses a color, or needs to be
re-encoded at a different frame rate.

## Checklist

- Use `ttyrec`, not `script`, for video-bound terminal captures.
- Unset `NO_COLOR` and force `TERM=xterm-256color` for Rich color.
- Pick `--stats`, `--display-events`, and `--display-tools` independently
  for the diagnostics you want in the live recording.
- Remember that one live owner renders all roles and `--record` saves after
  that owner render.
- Expect rapid live updates to coalesce to the latest frame; rely on the
  canonical/public output surfaces for lossless content.
- Preserve the `.ttyrec` file until the MP4 has been reviewed.
- Encode MP4 with `-pix_fmt yuv420p` and `-movflags +faststart` for broad
  playback compatibility.
