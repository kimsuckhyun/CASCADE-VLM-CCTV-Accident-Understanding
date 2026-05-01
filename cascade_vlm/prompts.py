#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for the VLM (coarse, refine, type-verify, grid).

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

from .config import TYPE_LIST


ACCIDENT_TYPE_5_DEFINITION = """
Accident type classification (MUST choose exactly one label):

1) single
- single-vehicle accident
- vehicle hits wall, guardrail, barrier, divider, pole, sign, curb, roadside structure
- rollover, flip, falling over, skid then crash, spin then crash, run-off-road crash
- NO direct collision with another vehicle at the first impact moment
- If there is visible vehicle-to-vehicle contact, DO NOT classify as single

2) rear-end
- one vehicle hits the back of another vehicle
- SAME-DIRECTION collision where the striking vehicle first contacts the rear area of the leading vehicle
- Both vehicles must be traveling in approximately the SAME direction

3) t-bone
- side impact at approximately right angle
- one vehicle first hits the SIDE of another vehicle
- commonly at an intersection where vehicles approach from PERPENDICULAR directions
- If vehicles come from cross-streets or one enters from a side road: this is t-bone, NOT rear-end

4) sideswipe
- side-to-side scraping or glancing contact
- vehicles first contact along their sides, often during lane change or merge
- shallow-angle side contact rather than direct perpendicular impact

5) head-on
- front-to-front collision
- two vehicles moving TOWARD each other first collide at the front

Critical anti-confusion rules:
- If vehicles approach from perpendicular directions (e.g., intersection): t-bone, NOT rear-end
- rear-end requires SAME direction travel
- If the frames show a vehicle physically contacting another vehicle at first impact, you MUST choose one of:
  "rear-end", "t-bone", "sideswipe", "head-on"
- In that case, "single" is forbidden.
- Use "single" only when the first crash is NOT with another vehicle.
"""

VEHICLE_ACCIDENT_DEFINITION = """
Vehicle accident definition (strict, first-contact based):

A valid accident must be a REAL vehicle accident occurring on or immediately adjacent to the drivable road area
(lanes, ramp, intersection, shoulder, median edge, roadside barrier zone).

Count as accident ONLY if one of these happens:

1) Vehicle-to-vehicle collision:
- rear-end collision
- side impact / sideswipe with real contact
- head-on collision
- angled collision
- merge or cut-in collision
- two or more vehicles physically contacting each other

2) Vehicle-to-road-structure collision:
- vehicle hits wall / guardrail / barrier / divider / curb / pole / sign structure

3) Single-vehicle accident:
- vehicle spins out and then physically crashes
- vehicle skids and then physically crashes
- vehicle rolls over / flips / falls over
- vehicle runs off the road and crashes

Important classification rule:
- If another vehicle is visibly involved in the FIRST physical contact, the accident is NOT single.
- Do NOT use single for vehicle-to-vehicle collisions.

Important event-time rule:
- peak_sec = the FIRST physical accident moment
- vehicle-to-vehicle: first frame where the vehicles first touch
- vehicle-to-structure: first frame where the vehicle first touches the structure
- single-vehicle: first frame where the event becomes a real crash, not just unstable motion

Important point rule:
- accident_point = the FIRST physical contact point
- NOT a bounding box
- NOT a vehicle center point
- NOT the brightest point
- NOT the frame center unless the true first contact is there
- Return exactly one pixel point {x, y}

Do NOT count as accident:
- near miss without contact
- normal lane change
- normal braking / slowing / stopping
- ordinary turning on a curved road
- a vehicle merely appearing in the frame
- a vehicle entering from off-screen
- headlights, reflections, shadows, water droplets on lens, snow streaks, rain streaks
- unrelated background motion
"""

LOW_VISIBILITY_RULES = """
Low-visibility reasoning rules:
- In dark/night scenes, do NOT rely on brightness flashes alone.
- In rain/snow scenes, do NOT treat reflections, glare, streaks, spray, or wet-road shine as accident evidence by themselves.
- In low-quality CCTV, prioritize temporal evidence across consecutive frames over a single blurry frame.
- Strong evidence includes abrupt trajectory change, first contact, tiny motion discontinuity, sudden stop after unstable motion, rotation/spin, rollover/crash posture, and consistent pre/post-impact behavior.
- Weak evidence alone is NOT sufficient: headlights getting brighter, normal brake lights, lane curvature on ramps, weather artifacts, compression noise, or a newly appearing vehicle.
"""

LENS_OBSTRUCTION_RULES = """
Lens obstruction / blur / droplet rules:
- CCTV footage may be blurred by low resolution, dirty lens, fogged dome, water droplets, smearing, or partial occlusion on the camera cover.
- A water droplet on the CCTV lens can create a soft circular blur, local bright distortion, warped highlight, or stationary translucent blob.
- Do NOT mistake lens droplets, lens blur, haze, or smear for a vehicle, accident point, or collision flash.
- If the image is partially blurred by lens obstruction, infer the crash from visible motion before/after the obscured area only when temporally consistent.
- A stationary blur patch on the lens is not an accident point.
- Prefer the true road interaction point over optical artifacts on the lens.
"""

DISTANT_SMALL_ACCIDENT_RULES = """
Small / distant accident rules:
- The true accident may happen far from the camera and may occupy only a very small number of pixels.
- Do NOT ignore a region just because the vehicles are tiny.
- Do NOT prefer large nearby vehicles over a small distant true crash.
- Do NOT prefer the image center just because it is visually salient.
- The first-contact point may be a tiny region near the horizon, near lane convergence, or near a far intersection.
- If direct contact is hard to resolve, infer the most likely first-contact point from consecutive-frame motion:
  approach -> convergence/contact -> sudden stop / deviation / spin / separation / blockage.
- A tiny but temporally consistent crash cue is better than a large but irrelevant nearby object.
- The correct point should be where the crash physically starts, even if very small.
"""

FALSE_POSITIVE_RULES = """
Common false positive patterns to AVOID:
- Tow truck / recovery vehicle arriving at scene: this is AFTER the accident, not the accident itself.
- Pedestrian or cyclist passing through: not a vehicle accident unless a vehicle hits them.
- Vehicle emerging from under a bridge/overpass: normal traffic, not an accident.
- Compression artifact / frame glitch / sudden video quality drop: camera issue, not impact.
- Red/bright colored vehicle just driving normally: color alone is not evidence.
- Vehicle headlights getting brighter as it approaches camera: normal approach, not collision.
- Snow/rain causing vehicles to slow down: slowing is not crashing.
- A vehicle entering the frame from off-screen: entry is not collision.
- Vehicles stopped at a red light or in traffic: normal stop, not crash.
- Construction or road maintenance equipment: not an accident.
- Emergency vehicle lights (ambulance, police): response to accident, not the accident moment itself.
"""

SCORE_CALIBRATION = """
Score calibration guide (CRITICAL — follow strictly):
- 0-15:  No vehicles or completely normal traffic. Nothing unusual at all.
- 16-30: Something slightly unusual (braking, lane change) but almost certainly NOT an accident.
- 31-50: Suspicious motion but unclear — could be a near-miss, aggressive driving, or minor incident.
- 51-70: Probable accident — clear trajectory disruption, abrupt stop, or likely contact visible.
- 71-85: Strong accident evidence — vehicles clearly colliding, spinning, or crashing.
- 86-100: Definite first-impact moment — unambiguous physical collision visible in the frames.

IMPORTANT CALIBRATION RULE:
- Most regions/time segments show normal traffic and should score BELOW 30.
- Only a small number of regions should exceed 60.
- If you see no clear physical collision evidence in the LOCAL region, score BELOW 30.
- Use the full-context panel only to understand where the local crop sits in the whole frame.
"""

SYSTEM_PROMPT = f"""You are an expert CCTV vehicle accident analyst.

{VEHICLE_ACCIDENT_DEFINITION}

{ACCIDENT_TYPE_5_DEFINITION}

{LOW_VISIBILITY_RULES}

{LENS_OBSTRUCTION_RULES}

{DISTANT_SMALL_ACCIDENT_RULES}

{FALSE_POSITIVE_RULES}

Rules:
- The video contains exactly one true vehicle accident event.
- Use only the provided frame timestamps.
- Frames are provided in CHRONOLOGICAL ORDER. Each frame has a label "[N/Total] T.TTs".
- Some images show FULL CONTEXT with the local region highlighted; some show the LOCAL CROP.
- Use the LOCAL CROP to judge small/distant events, and use FULL CONTEXT only as reference.
- Output ONLY one valid JSON object.
- Do NOT output reasoning, markdown, or any text before/after the JSON.
"""


# ─────────────────────────────────────────────
