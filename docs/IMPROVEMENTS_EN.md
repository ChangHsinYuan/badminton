# Technical Documentation — Calibration Principle · Methodology · Changelog (TECH)

The single technical document of this fork: court calibration principle and failure
criteria, shot-type confidence, design methodology, feature evolution (changelog),
and known limitations. For usage, see the [README](../README.md) (Chinese) or
[README_EN](../README_EN.md).

## Table of Contents

- [Court Calibration Principle and Failure Criteria](#court-calibration-principle-and-failure-criteria)
- [Shot-Type Confidence: Impact of Missing Ball Trajectories](#shot-type-confidence-impact-of-missing-ball-trajectories)
- [The 2D Boundary and the Pseudo-3D Route](#the-2d-boundary-and-the-pseudo-3d-route)
- [v1: AI Coach Analysis (2026-11-14)](#v1-ai-coach-analysis-2026-11-14)
- [v2.1: Shot Type Recognition (2026-11-16)](#v21-shot-type-recognition-2026-11-16)
- [v2.2: Pro Style Feature-Card Matching (2026-11-16)](#v22-pro-style-feature-card-matching-2026-11-16)
- [Cross-Video Validation (2026-11-16)](#cross-video-validation-2026-11-16)
- [Methodology Principles (2026-11-16)](#methodology-principles-2026-11-16)
- [Known Limitations and Future Directions](#known-limitations-and-future-directions)

---

## Court Calibration Principle and Failure Criteria

### It is not "solving equations" — it is "model matching + hypothesis enumeration"

A homography has 8 degrees of freedom; solving equations from line segments alone
is not enough (`court_fit.py`). The actual mechanism:

1. White-line mask → Hough transform extracts segments → collinear segments merged
   into **infinite-line parameters** (a,b,c) — a partially visible segment does not
   hurt the estimate of the whole line
2. Lines are split into two families by orientation (sideline family / baseline
   family, corresponding to the court's two orthogonal directions)
3. **Correspondence-hypothesis enumeration**: take 2 lines per family → 4
   intersections, pair them against combinations of known court-line coordinate
   values (sidelines x∈{0, 0.46, 3.05, 5.64, 6.1}, baselines
   y∈{0, 0.76, 4.72, 6.7, 8.68, 12.64, 13.4}); each correspondence yields a
   candidate H via 2×2-intersection DLT — **we never need to know which detected
   line is which model line**: all combinations are enumerated and the score decides
4. Score and select: bidirectional Chamfer (projected model lines must hug white
   lines + white lines must be explained by projected lines) + **court-coordinate
   consistency** (each detected line, mapped back to court coordinates, must be a
   constant-x or constant-y line close to some model line value) → best hypothesis
   refined by 8-DOF Nelder-Mead + subset-ICP correspondence expansion

The "two lines" intuition: 2 lines per family give 4 intersections = 8 constraints,
exactly pinning the 8 DOF — that is the minimum configuration; in practice a dozen
lines join the scoring, and robustness comes from redundancy.

### Incomplete lines are the norm — triple tolerance

- Hough lines are whole-line parameters; a missing segment does not affect the estimate
- Scoring is per-line, per-sample-point; missing segments are skipped automatically
  (score drops, nothing breaks)
- Far-side lines often survive only as short stubs at low angles: "subset ICP" uses
  tight matches as the base and tries loose candidates subset by subset, preventing
  bad lines from dragging down the full DLT

### When it cannot (failure criteria)

- **Hard failure** (explicit error "no valid homography hypothesis found"): fewer
  than 2 lines in one family (e.g., only the net line visible); the two families
  nearly parallel (intersections outside the image are filtered out)
- **Soft failure** (a solution exists but don't trust it): total score notably low;
  `per_line` support BAD and concentrated on the far side (hints lens distortion —
  homography assumes pinhole + plane; barrel distortion systematically displaces
  far lines); adjacent-court lines leaking into the mask cause wrong snapping
  (coverage < 1, white points map outside the court)
- **Final criterion**: human check of `fit_overlay.jpg` — do projected lines sit
  on the white lines?
- **Mirror ambiguity**: every badminton court line is symmetric about the center
  line x=3.05 (the net line included), so a left-right mirrored solution exists
  with an identical score — the lines themselves cannot disambiguate, and both
  solutions "hug" the overlay. The impact is limited to swapped left/right zone
  labels; speed / net distance / turn rate / racket-hand voting (image coordinates)
  are unaffected. Disambiguating clues: service court rules, a scoreboard, or a
  one-time manual check on the overlay

## Shot-Type Confidence: Impact of Missing Ball Trajectories

Missed ball detections are currently the **weakest link** and introduce systematic bias:

- **Mitigations in the mechanism**: arcs with <4 points (MIN_PTS) or <0.12 s are not
  classified at all — prefer "unrecognized" over guessing; classification thresholds
  are strict (a smash requires apex position <0.25 AND convexity <0.13 AND descent
  peak >600 px/s — precision over recall)
- **Systematic bias**: YOLO detects slow, high arcs best (bigger ball, longer
  trajectory, unoccluded); fast shots (smashes/drives) suffer from motion blur and
  large inter-frame displacement; short net balls are often occluded by bodies.
  A report's "82–84% clears/lobs" should be read as the distribution **among
  detected arcs**, not the true shot distribution — slow shots being easy to detect
  is itself selection bias, and smashes are systematically undercounted (both
  validation videos showing ~83% clears is exactly the suspicious signal)
- **Concrete damage from missing points**: convexity/apex features need the arc top
  visible — the ball is fastest and most easily dropped at its apex; once the top is
  lost, the "clear convexity >0.30" rule fails, falling into unrecognized or a drop misclassification
- **Usage advice**: read shot distributions as relative tendencies between the two
  players (who approaches the net more), not as absolute shares; treat smash counts
  as a lower bound of the code
- **Improvement directions**: TrackNetV3 for detection rate; short-gap interpolation
  (never across a hit — it would merge two arcs into one); per-class detectability
  priors to de-bias; **fusing hand swing trajectories** (wrist detection rate is far
  higher than the ball's and continuous — the most promising substitute signal)

## The 2D Boundary and the Pseudo-3D Route

Current processing has two layers — one correct 2D, one compromised 2D:

- **The players' 2D is the correct model**: feet are on the ground, so the
  ground-plane homography is mathematically lossless; metric speed (m/s),
  distances, and positioning are true physical quantities
- **The ball's 2D is a compromise**: only image (u,v) and pixel speeds exist;
  **height information is entirely absent**. This is exactly why the reach metric
  was deleted (projecting a wrist 1 m above ground as a ground point → median
  error 10 m+); it is also why shot classification uses only image-geometric
  features (convexity/apex/pixel speed) with thresholds tied to the
  "low-angle 720p/30fps" calibration — strictly speaking, thresholds need
  re-tuning for a different camera setup
- **Design boundary**: image arc features are sufficient for **discrete
  classification** ("which shot is this"); they cannot yield **physical quantities**
  (shuttle speed in m/s, apex height). The pipeline deliberately stops at classification
- **Pseudo-3D route**: a monocular low-angle camera has no intrinsics, so strict 3D
  is out of reach — but the net height of 1.55 m is known: the net top lies on both
  the ground plane and the 1.55 m-height plane, and the two-plane homography
  combination can solve the vertical vanishing point, decoupling the ball's height
  from its depth, recovering "how high and how far the ball is". True 3D requires
  stereo or multiple views

---

## v1: AI Coach Analysis (2026-11-14)

**New modules**: `style_features.py` / `keyframes.py` / `style_coach.py`

Extracts footwork/positioning/rhythm features from `detections.jsonl`, renders key
frames (sprint / swing / ball contact) with skeletons, and has an LLM produce a
five-section report (style portrait / positioning & footwork / pro-style similarity /
training advice / key-frame review).

**Pitfall log**:

1. **Depth-zone coordinate flip (severe)**: the first version treated y=0 as the net;
   actually y=0 is the far baseline and y=6.7 is the net. The report flipped
   baseline-oriented players into "net spiders" — the whole narrative inverted.
   Fix: all depth zones now use `dist_from_net()` (distance to net); upper uses
   `6.7-y`, lower uses `y-6.7`.
   **Lesson: any statistic involving front/back-court semantics must first declare
   its coordinate convention and cross-check against the home position (the mean of
   the heat-zone distribution should match the home position).**
2. **Double-transform bug**: the zone function already converted coordinates, but
   the call site passed converted values again — all zone results wrong.
3. **RTMPose low-confidence keypoint pollution**: hand keypoints output (0,0) when
   confidence is low, polluting reach/turn statistics; must filter before computing.
4. **Reasoning-model output pollution**: glm-5.3-flash's chain-of-thought leaked into
   the answer and got truncated by max_tokens; text analysis now defaults to
   qwen3.8-chat, overridable via `--model`.

## v2.1: Shot Type Recognition (2026-11-16)

**New module**: `shot_types.py` — ball trajectory segmentation + rule-based
classification (smash / drop / clear / drive).

### Method

1. Detection points are split into runs by frame gaps → hit boundaries detected by
   the primary criterion "strong upward motion after the hit" with a "sudden speed
   jump" fallback, then runs merged
2. Per-arc features: duration / convexity ratio / apex position / peak descent
   speed / mean speed / peak 2D speed (all in image space)
3. Rule-based classification; landing zones are labeled with **the next hitter's
   court position** (ground truth), completely bypassing the unreliability of
   projecting an airborne ball onto the ground

### Calibration process (low-angle 720p/30fps Lin Dan Cup footage)

| Round | Problem | Fix |
|---|---|---|
| 1 | Dense-grid interpolation stitched large gaps into many fake arcs, fake speeds up to 540k px/s | Dropped interpolation for run merging; speeds only from real adjacent-frame diffs |
| 2 | Wrong mapper import path (`court.mapper` → actually `badminton_analysis.court.mapper`) made landings all empty | Fixed import; a homography ground-projection of an airborne ball always flies out of court — switched to receiver position |
| 3 | Descent-speed sign inverted (image y points down); drives/smashes all wrong | Unified "positive descent speed" convention |
| 4 | Vision-API labeling of 24 samples: only 19% agreement, exposing idle segments mixed in (5 px/s ball-pickup waits) and an overly tight drive-duration gate | Idle filter ms<60; drive dur<1.5; widened drop convexity range; peaks take the second-largest value to resist single-frame jitter |
| 5 | Fragmentation: any sign flip broke an arc | Primary hit criterion changed to "strong upward motion on the outgoing side" (incoming-side speed may be small; vertical velocity naturally approaches zero at contact) |

### Known limitations (told honestly to users)

- **Limited coverage**: at 45% ball detection rate, a 90-second rally yields ~19
  recognized shots (of ~64 estimated total hits); distributions are reference only.
  The prompt already states "do not make claims about individual shots"
- **Depth-height aliasing**: in monocular image space, vertical motion mixes with
  depth motion (a ball flying toward the camera also "falls" in the image) — the
  inherent precision ceiling of monocular low angles; a full fix requires 3D ball
  trajectories or stereo
- **Ownership perspective bias**: the same physical distance maps to more pixels
  for the near player; the ownership distance gate was relaxed to 420 px. Reports
  output the "rally alternation rate" as an ownership-quality QA metric (ideal for
  singles ~90%+; measured 45% on this video — ownership is noisy)

## v2.2: Pro Style Feature-Card Matching (2026-11-16)

**New modules**: `pro_profiles.py` + the `profiles/` card library.

### Motivation

v1's "pro style similarity" let the LLM free-associate from training memory — not
reproducible. Measured case: the near player in the Lin Dan Cup video is Momota
Kento himself, yet the LLM compared him to Viktor Axelsen and compared the far
amateur to Momota — **completely mismatched**.

### Method

- Feature card = normalized feature vector (4 style indices + 3-zone positioning
  shares + speed/sprint/turn + shot-type distribution)
- Similarity = weighted L2 distance (weights in `pro_profiles.SPEC`); missing
  features are skipped automatically and weights re-normalized
- `--build` creates a measured card from any video; `--match` computes nearest
  neighbors against all cards in the library
- The LLM only **interprets computed results** (why similar / where the differences
  are); it no longer invents comparisons itself

### Validation

After building a Momota Kento card from this video and re-matching: near player →
Momota card **100%** self-match, cross-match 86% — good discrimination. The LLM
interpretation also rehabilitated "turn rate 0.06 /s": the v1 report listed it as
the near player's weakness, while the v2 report, combined with Momota's identity,
correctly reads it as "extremely fluid footwork with no violent direction changes —
exactly what technical mastery looks like".

### Notes

- The two existing cards (`source=measured`) were measured from this same video —
  a trivial self-match; only cross-video comparison is meaningful
- The library ships 6 star style profile cards (Lin Dan / Lee Chong Wei / Chen Long /
  Axelsen / Ginting / Antonsen, `source=curated`): parameters are **estimates** from
  public style material (Wikipedia playing-style descriptions cross-checked with LLM
  knowledge), order-of-magnitude calibrated to this pipeline's feature scale, but
  not measured from the stars' actual match videos; style-tendency reference only,
  excluded from precise claims
- Measured validation: the closest star card for both players in this video is Chen
  Long (80%, defensive counter-attacking), consistent with the measured slow-paced,
  clear-heavy profile; explosive attackers (Ginting / Axelsen / Lin Dan) rank lower —
  discrimination as expected

## Integration and Engineering Improvements (2026-11-16)

- `style_coach.py` adds `--player-hint "lower=Momota Kento,upper=Lin Dan"`: known
  identities feed the LLM directly, turning the report from "guessing styles" into
  "verifying data against real players"
- `config/llm_api.json` timeout 120→300 (gateway read timeouts occasionally with
  long prompts)
- Chinese fonts: matplotlib registers system Noto / AR PL fonts, fixing DejaVu
  missing-glyph warnings

## Racket-Hand Misjudgment Fix (2026-11-16)

**Problem**: the v1 report called both players "right-handed"; the near player is
actually left-handed (user confirmed by watching the video).

**Causes** (two layers):
1. The original heuristic "which wrist was detected in more frames" has nothing to
   do with handedness — detection-rate differences come from occlusion and
   confidence (measured: near player's left/right wrists detected in 1647/2351
   frames, yet he plays left-handed)
2. The first fix's "Top-K frames where the ball is closest to either hand across the
   whole video" voting also errs: frames where the ball **flies past** the body cast
   fake votes (the near player's 5 closest frames all favored the right wrist — but
   those were fly-bys, not hits)

**Fix** (`style_features.infer_hand_pref`):
- The primary signal is now **hit-frame voting after shot_types ownership**: for each
  shot, within the frame0-4~frame0+1 window take the frame where both hands are
  closest to the ball and cast one vote (within the window the ball must belong to
  this exchange, so no close-distance gate — at frame0 the ball has often already
  flown 200–400 px away); only when hit votes < 5 does it fall back to the whole-video
  Top-K (with a 180 px gate)
- Votes count only when both hand keypoints are visible in the same frame, avoiding
  single-side occlusion bias; a conclusion requires ≥60% majority and ≥3 votes
- The report annotates the votes: `Racket hand: left (proximity vote left 10 : right 5)`
- Result: far player right-handed (4:20), near player left-handed (10:5), matching
  human review
- Identity corroboration: near player `--player-hint "lower=Momota Kento"` — Momota
  is himself left-handed (English Wikipedia handedness: Left); hand detection,
  identity, and user observation all agree

**Incidental fix**: the reach metric (`reach_mean_m`) turned out to be noise — wrists
sit 1 m+ above ground, so the ground-homography projection throws them far behind
(median error 10 m+); old values came from a few dozen stray samples per year that
happened to fall into the 0.3–2.5 m gate; multiplying image-space distance by the
ground depth scale at low angles also amplifies "height displacement" by the "depth
scale" (the mean was pinned at the clamp). Without camera intrinsics this cannot be
fixed, so the **metric was deleted** — honesty over fake precision.

## Cross-Video Validation (2026-11-16)

The full pipeline was re-run on an **unseen second video**, validating both the
fork's core selling point and the v2 feature cards:

**Material**: Wuhan Lin Dan Cup round of 16, Momota Kento vs Wang Yifan, low angle
from behind the court at courtside, a 9000-frame clip (downloaded with
`fetch_video.sh` segment mode; found via a Bilibili wbi-signed search script and
manually curated — official "HD replays" are usually multi-camera broadcasts and
unusable; fixed low-angle uploads from content creators are the target material)

**Automatic calibration reproduction**: `fit_court_driver.py` scored 0.796 (same
level as the old video), the blue net line perfectly on the line, green lines
slightly off but usable for coordinate conversion — "any courtside low-angle video
works" with zero parameter tuning

**Cross-video identity (style fingerprint)**: the new video's near player matches
the `Momota Kento (measured)` card built from the old video at **96.1%** — the first
non-self-match identity validation: the same player (Momota) appears in two videos
from different camera positions, and the style fingerprint recognized him (turn rate
0.12 vs 0.125 /s, midfield-control type, etc.). Player identity is now defined by
style matching, not by seeing the picture clearly.

**Racket-hand inheritance**: on the new video, proximity voting misjudged the near
player (Momota, a lefty) as right-handed (left 12 : right 22), while the same code
was correct on the old video — the proximity-vote signal **depends on camera
geometry and is unstable across camera positions** (the extension vote flipped the
same way). Fix: `--build` now stores `hand_pref` in the card; when generating a
report, if a side matches a measured card at ≥90%, the racket hand is **inherited
from the card** and the summary annotates the source and local votes:
`Racket hand: left (inherited from matched card Momota Kento; this video's proximity
vote left 12 : right 22)` — a player's racket hand does not change, so under a
high-confidence identity match the card is more trustworthy than a single-video
vote. When the inherited value agrees with the local vote, the display is unchanged
(old-video output unaffected).

## Methodology Principles (2026-11-16)

Four design principles spanning v1–v2.2:

1. **Separate computation from interpretation**: the LLM never computes anything.
   All numbers (style indices / similarities / votes) are computed by deterministic
   code and injected into the prompt as "computed result blocks" with the
   instruction "interpret only, never invent" — plugging the biggest disease of
   LLM analysis tools: making up data.
2. **Style as identity**: at low angles faces/jerseys are unreadable (the vision
   model was observed guessing wildly), so motion-statistic features serve as the
   player's ID card — the same body keeps its style across camera positions; it is
   a more essential feature than appearance. Identity is defined by feature-card
   nearest neighbors (Momota card 96.1% cross-video), not by seeing the picture.
3. **Traceable evidence chain + honesty first**: every conclusion carries its source
   (`inherited from matched card X; this video's votes a:b`); noisy metrics are
   deleted (reach) rather than force-fixed; curated cards are labeled and excluded
   from precise claims; a 100% self-match is a trivial solution and does not count
   as validation — only cross-video does.
4. **Signal-confidence layering**: redundant signals + explicit failure modes.
   Racket hand as the example: proximity voting depends on camera geometry (the
   same code flipped on a new camera), so an inheritance chain was built — the card
   record of a high-confidence identity match > a single-video vote (a player does
   not change hands; votes flip with camera angles).

## v3: Monocular 3D Measurement Chain (2026-11-18)

Three steps recover metric physical quantities from 2D video, using only court-line
self-calibration — no calibration objects:

1. **Camera self-calibration** (`camera_selfcalib.py`): decompose the court
   homography inv(H) into K/R/t (centered principal point + square-pixel
   assumption, two equations solved for f and averaged). **Left-handed trap**:
   the mirror canonicalization makes the H frame left-handed; the orientation
   class must be auto-selected by requiring camera height C_z > 0, otherwise the
   camera lands below the floor; downstream 3D points need y_w = 13.4 − y_hn.
   Camera positions calibrated independently on two videos agree within 12 cm;
   net-top (1.55 m) projection vs the actual tape measures ±1–7 px.
2. **Contact-point heights** (`hit_points3d.py`): closest approach between the
   ball ray and the vertical line through the hitter's foot position. Clear/lift
   contact points median 2.2–2.4 m; upper/lower gap 0.8–1.0 m ≈ arm + racket,
   dimensionally self-consistent.
3. **Physics trajectory fitting** (`trajectory3d.py`): RK4 with gravity +
   quadratic drag, depth-grid initialization + two-stage refinement; a strong
   receiver anchor (by the end the ball has slowed to ~7 m/s and must be within
   1.5 m of the opponent) plus a weak hitter anchor break the monocular
   "slow-near / fast-far" speed–depth degeneracy. β converges to 0.19–0.21
   (physical truth ≈0.22, shuttle terminal velocity 6.7 m/s); the new video
   fits 119 arcs with 0 failures, top speed 238 km/h (back-extrapolation only
   0.05 s = first detection is the contact, high confidence), clear/lift shot
   speed median 24.2 m/s, net clearance +0.4 to +0.9 m.

**Validation logic**: the drag coefficient β is the physics-level touchstone of
the whole chain — if calibration, anchors, or the time axis were wrong anywhere,
β would not settle near 0.2; its convergence certifies the chain. `visualize3d.py`
draws the trajectories back onto the video (yellow = trusted, gray + ?? =
low-confidence) for eyeball verification; the README top shows the demo clip.

**Known boundaries**: for fast shots the first detection lags the actual contact,
so shot speed is physically back-extrapolated (`speed_at_owner_axis`);
t_back > 0.3 s is unreliable (json stores t_back for filtering). Detection
misses (motion blur / occlusion) cap coverage — every drawn trajectory is fitted
from real detections; interpolation only fills geometric gaps.

## Known Limitations and Future Directions

1. **Ball detection**: TrackNetV3 (32-channel input) can improve detection rate and
   precision → directly improves shot-type coverage
2. **3D ball trajectory**: implemented by the v3 monocular 3D measurement chain
   (self-calibration + anchored fitting, above); remaining: specificity validation
   of the far-player style cards on the 3D chain, generalization to more camera
   positions, higher frame rate (30 fps caps temporal precision)
3. **Card library expansion**: batch `--build` on public pro match videos to build a
   cross-player comparable card library
4. **Shot types**: fuse hand swing trajectories (hands data already exists) with the
   ball trajectory, replacing pure ball-trajectory rules
5. **Turn-rate metric**: systematically underestimates "energy-saving" footwork;
   reports should interpret it together with identity/style, not as a standalone weakness
6. **Ball detection bias → biased shot distribution**: see
   [Shot-Type Confidence](#shot-type-confidence-impact-of-missing-ball-trajectories) above
7. **Metric ball quantities bounded by detection precision**: v3 broke the "pure 2D,
   no physical quantities" boundary, but shot-speed / contact-point accuracy is
   still constrained by detection noise (see v3 known boundaries)
8. **Calibration left-right mirror ambiguity**: see
   [Court Calibration Principle and Failure Criteria](#court-calibration-principle-and-failure-criteria) above
