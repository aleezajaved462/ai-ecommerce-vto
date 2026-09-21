
"""
Virtual Try-On Socket.IO server

Key features in this version:
  1. TORSO items (apparel/tshirt/jacket) are now rendered with a PERSPECTIVE
     WARP: the garment's 4 corners are mapped onto the person's actual
     shoulder-left, shoulder-right, hip-right, hip-left points every frame.
     This means the shirt stretches/shrinks to match each person's real
     shoulder width, hip width, and torso length -- it follows body shape
     instead of being one rigid rectangle scaled by a single number.
  2. WRIST items (watch) and FACE items (eyewear/cap) still use the simpler
     resize+rotate+paste approach, which is appropriate for small rigid
     accessories.
  3. EMA smoothing (per-point for torso quads, per-value for wrist/face) so
     nothing jitters.
  4. When pose detection misses a frame, the last known placement is held
     for a short window instead of the overlay disappearing/blinking.
  5. Alpha channel validation + auto-trim of transparent borders.
  6. reload=False so the uploaded overlay isn't lost on server auto-restart.
  7. Smart resize (LANCZOS4 for upscaling, AREA for downscaling) for sharper
     wrist/face overlays.

Run:  python main.py
Deps: pip install python-socketio uvicorn opencv-python pillow pillow-avif-plugin mediapipe numpy
"""

import base64
import io
import math

import cv2
import numpy as np
import socketio
import uvicorn
from PIL import Image

try:
    import pillow_avif  # noqa: F401  (registers AVIF format support for Pillow)
except ImportError:
    print("[warn] pillow-avif-plugin missing - AVIF images will not be supported")

import mediapipe as mp

# ---------------------------------------------------------------- config

# Placement tuning per product category.
#
# For anchor == "torso" (apparel/tshirt/jacket), the garment is perspective-
# warped onto a quad built from shoulder + hip landmarks:
#   shoulder_width_mult : garment width at the shoulder line, as a multiple
#                          of the detected shoulder-to-shoulder distance
#   hip_width_mult       : garment width at the hem line, as a multiple of
#                          the detected hip-to-hip distance (controls flare)
#   top_offset            : how far above the shoulder line the collar sits,
#                          as a ratio of shoulder width (for neckline/collar)
#   bottom_offset         : how far past the hip line the hem extends,
#                          as a ratio of torso length (controls shirt length)
#
# For anchor == "wrist" / "eyes" / "head", the old width_mult / y_offset
# scheme is used with simple resize+rotate+paste.
PRODUCT_CONFIG = {
    "apparel": {
        "anchor": "torso",
        "shoulder_width_mult": 1.20,
        "hip_width_mult": 1.05,
        "top_offset": 0.12,
        "bottom_offset": 0.18,
    },
    "tshirt": {
        "anchor": "torso",
        "shoulder_width_mult": 1.20,
        "hip_width_mult": 1.05,
        "top_offset": 0.12,
        "bottom_offset": 0.18,
    },
    "jacket": {
        "anchor": "torso",
        "shoulder_width_mult": 1.35,
        "hip_width_mult": 1.20,
        "top_offset": 0.16,
        "bottom_offset": 0.22,
    },
    "eyewear": {"anchor": "eyes", "width_mult": 2.10, "y_offset": 0.00},
    "cap":     {"anchor": "head", "width_mult": 2.40, "y_offset": -0.55},
    "watch":   {"anchor": "wrist", "width_mult": 0.40, "y_offset": 0.0},
}

SMOOTHING = 0.65          # 0 = no smoothing, 0.9 = very slow/laggy. 0.6-0.75 is a good range.
HOLD_FRAMES = 30          # frames to hold last known placement after detection is lost
MIN_VISIBILITY = 0.30     # landmark visibility threshold (lower = more forgiving, less blinking)
POSE_INPUT_WIDTH = 480    # downscale width used for pose detection (for speed)
JPEG_QUALITY = 90

# ---------------------------------------------------------------- state

current_overlay_img = None      # BGRA numpy array
current_product_type = "apparel"

# Per-client smoothing state.
# For wrist/face: sid -> {"cx","cy","w","angle","miss"}
# For torso quads: (sid + "_quad") -> {"pts": np.float32 shape (4,2), "miss": int}
track_state = {}

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = socketio.ASGIApp(sio)

# ---------------------------------------------------------------- helpers


def trim_alpha(img):
    """Trims transparent borders so the product image aligns tightly to its own edges."""
    if img is None or img.shape[2] < 4:
        return img
    alpha = img[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if ys.size == 0 or xs.size == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def decode_image_bytes(b64_string):
    """Decodes a Base64 string to a BGRA matrix (supports PNG / WEBP / AVIF / JPG)."""
    try:
        encoded = b64_string.split(",", 1)[1] if "," in b64_string else b64_string
        pil_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
        bgra = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        bgra = trim_alpha(bgra)

        if bgra[:, :, 3].min() == 255:
            print("[warn] This image has no transparency (likely a JPG). "
                  "A solid box will appear on the body - use a transparent PNG/WEBP instead.")
        print(f"[ok] Overlay loaded: {bgra.shape[1]}x{bgra.shape[0]}")
        return bgra
    except Exception as e:
        print(f"[error] decode_image_bytes: {e}")
        return None


def cv2_to_base64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def smart_resize(img, target_w, target_h):
    """Chooses the right interpolation based on whether we're enlarging or shrinking."""
    target_w, target_h = max(1, target_w), max(1, target_h)
    src_h, src_w = img.shape[:2]
    if target_w * target_h > src_w * src_h:
        interp = cv2.INTER_LANCZOS4   # upscaling -> sharper result
    else:
        interp = cv2.INTER_AREA       # downscaling -> best quality
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def rotate_rgba(img, angle_deg):
    """Rotates an RGBA image against a transparent background (no cropping)."""
    if abs(angle_deg) < 0.8:
        return img
    h, w = img.shape[:2]
    diag = int(math.hypot(w, h)) + 2
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += (diag - w) / 2.0
    M[1, 2] += (diag - h) / 2.0
    return cv2.warpAffine(
        img, M, (diag, diag),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def alpha_paste(frame, overlay, cx, cy):
    """Alpha-blends a small overlay onto the frame centered at (cx, cy). Edge-safe."""
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]

    ox, oy = int(round(cx - ow / 2.0)), int(round(cy - oh / 2.0))

    x1, y1 = max(0, ox), max(0, oy)
    x2, y2 = min(fw, ox + ow), min(fh, oy + oh)
    if x2 <= x1 or y2 <= y1:
        return frame

    sx1, sy1 = x1 - ox, y1 - oy
    crop = overlay[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]

    if crop.shape[2] == 4:
        alpha = (crop[:, :, 3:4].astype(np.float32)) / 255.0
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        blended = alpha * crop[:, :, :3].astype(np.float32) + (1.0 - alpha) * roi
        frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = crop[:, :, :3]
    return frame


def alpha_blend_full(frame, warped):
    """Alpha-blends a full-frame-sized warped RGBA image onto the frame."""
    if warped.shape[2] != 4:
        return frame
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    frame_f = frame.astype(np.float32)
    blended = alpha * warped[:, :, :3].astype(np.float32) + (1.0 - alpha) * frame_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def _pt(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32), p.visibility


# ---------------------------------------------------------- torso (body-fit) logic


def compute_torso_quad(landmarks, w, h, cfg):
    """Builds a 4-point quad (top-left, top-right, bottom-right, bottom-left)
    from shoulder + hip landmarks, sized to THIS person's actual body
    proportions. Returns None if landmarks aren't visible enough.
    """
    ls, v1 = _pt(landmarks, PL.LEFT_SHOULDER, w, h)
    rs, v2 = _pt(landmarks, PL.RIGHT_SHOULDER, w, h)
    if min(v1, v2) < MIN_VISIBILITY:
        return None

    shoulder_mid = (ls + rs) / 2.0
    shoulder_w = float(np.linalg.norm(ls - rs))
    if shoulder_w < 12:
        return None

    lh, v3 = _pt(landmarks, PL.LEFT_HIP, w, h)
    rh, v4 = _pt(landmarks, PL.RIGHT_HIP, w, h)

    if min(v3, v4) >= MIN_VISIBILITY:
        hip_mid = (lh + rh) / 2.0
        hip_w = float(np.linalg.norm(lh - rh))
        torso_vec = hip_mid - shoulder_mid
    else:
        # Hips not visible (close-up shot) -> estimate proportionally to shoulder width
        torso_vec = np.array([0.0, shoulder_w * 1.5], dtype=np.float32)
        hip_mid = shoulder_mid + torso_vec
        hip_w = shoulder_w * 0.9

    torso_len = float(np.linalg.norm(torso_vec))
    if torso_len < 1e-3:
        return None

    down = torso_vec / torso_len                          # unit vector: shoulder -> hip
    across = np.array([-down[1], down[0]], dtype=np.float32)  # perpendicular: left <-> right

    shoulder_half = (shoulder_w * cfg["shoulder_width_mult"]) / 2.0
    hip_half = (hip_w * cfg["hip_width_mult"]) / 2.0

    top_center = shoulder_mid - down * (shoulder_w * cfg["top_offset"])
    bottom_center = shoulder_mid + down * (torso_len * (1.0 + cfg["bottom_offset"]))

    top_left = top_center - across * shoulder_half
    top_right = top_center + across * shoulder_half
    bottom_right = bottom_center + across * hip_half
    bottom_left = bottom_center - across * hip_half

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def smooth_quad(sid, quad):
    """EMA smoothing across all 4 quad points. Holds last quad briefly on miss."""
    key = sid + "_quad"
    st = track_state.get(key)

    if quad is None:
        if st is None:
            return None
        st["miss"] += 1
        if st["miss"] > HOLD_FRAMES:
            track_state.pop(key, None)
            return None
        return st["pts"]

    if st is None:
        track_state[key] = {"pts": quad.copy(), "miss": 0}
        return quad

    a = SMOOTHING
    st["pts"] = a * st["pts"] + (1 - a) * quad
    st["miss"] = 0
    return st["pts"]


def render_torso_overlay(sid, frame, overlay_img, cfg):
    """Perspective-warps the garment onto the detected body quad. Body-shape aware."""
    h, w = frame.shape[:2]

    scale = POSE_INPUT_WIDTH / float(w) if w > POSE_INPUT_WIDTH else 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = pose_detector.process(rgb)

    quad = None
    if result.pose_landmarks:
        quad = compute_torso_quad(result.pose_landmarks.landmark, w, h, cfg)

    print(f"[debug] torso detected={quad is not None}")

    quad = smooth_quad(sid, quad)
    if quad is None:
        return frame, False

    oh, ow = overlay_img.shape[:2]
    src_pts = np.float32([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]])

    M = cv2.getPerspectiveTransform(src_pts, quad)
    warped = cv2.warpPerspective(
        overlay_img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    frame = alpha_blend_full(frame, warped)
    return frame, True


# ---------------------------------------------------------- wrist / face logic


def _compute_wrist_placement(landmarks, w, h, cfg):
    """Anchors a product (watch/bracelet) to whichever wrist is more visible."""
    l_wrist, lv_w = _pt(landmarks, PL.LEFT_WRIST, w, h)
    r_wrist, rv_w = _pt(landmarks, PL.RIGHT_WRIST, w, h)
    l_elbow, lv_e = _pt(landmarks, PL.LEFT_ELBOW, w, h)
    r_elbow, rv_e = _pt(landmarks, PL.RIGHT_ELBOW, w, h)

    left_ok = min(lv_w, lv_e) >= MIN_VISIBILITY
    right_ok = min(rv_w, rv_e) >= MIN_VISIBILITY
    if not left_ok and not right_ok:
        return None

    if left_ok and (not right_ok or (lv_w + lv_e) >= (rv_w + rv_e)):
        wrist, elbow = l_wrist, l_elbow
    else:
        wrist, elbow = r_wrist, r_elbow

    forearm_vec = wrist - elbow
    forearm_len = float(np.linalg.norm(forearm_vec))
    if forearm_len < 8:
        return None

    angle = math.degrees(math.atan2(forearm_vec[1], forearm_vec[0])) - 90
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    width = forearm_len * cfg["width_mult"]
    return float(wrist[0]), float(wrist[1]), width, angle


def compute_face_placement(landmarks, w, h, cfg):
    le, v1 = _pt(landmarks, PL.LEFT_EYE, w, h)
    re, v2 = _pt(landmarks, PL.RIGHT_EYE, w, h)
    lear, v3 = _pt(landmarks, PL.LEFT_EAR, w, h)
    rear, v4 = _pt(landmarks, PL.RIGHT_EAR, w, h)
    if min(v1, v2) < MIN_VISIBILITY:
        return None

    eye_mid = (le + re) / 2.0
    if min(v3, v4) >= MIN_VISIBILITY:
        face_w = float(np.linalg.norm(lear - rear))
    else:
        face_w = float(np.linalg.norm(le - re)) * 2.6
    if face_w < 10:
        return None

    dx, dy = (re - le)
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    center = eye_mid + np.array([0.0, face_w * cfg["y_offset"]], dtype=np.float32)
    return float(center[0]), float(center[1]), face_w * cfg["width_mult"], angle


def smooth_point(sid, target):
    """EMA smoothing for (cx, cy, width, angle) tuples (wrist/face)."""
    st = track_state.get(sid)

    if target is None:
        if st is None:
            return None
        st["miss"] += 1
        if st["miss"] > HOLD_FRAMES:
            track_state.pop(sid, None)
            return None
        return st["cx"], st["cy"], st["w"], st["angle"]

    cx, cy, width, angle = target
    if st is None:
        track_state[sid] = {"cx": cx, "cy": cy, "w": width, "angle": angle, "miss": 0}
        return cx, cy, width, angle

    a = SMOOTHING
    d_ang = ((angle - st["angle"] + 180) % 360) - 180
    st["cx"] = a * st["cx"] + (1 - a) * cx
    st["cy"] = a * st["cy"] + (1 - a) * cy
    st["w"] = a * st["w"] + (1 - a) * width
    st["angle"] = st["angle"] + (1 - a) * d_ang
    st["miss"] = 0
    return st["cx"], st["cy"], st["w"], st["angle"]


def render_point_overlay(sid, frame, overlay_img, cfg, kind):
    """Rigid resize+rotate+paste, used for wrist and face accessories."""
    h, w = frame.shape[:2]

    scale = POSE_INPUT_WIDTH / float(w) if w > POSE_INPUT_WIDTH else 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = pose_detector.process(rgb)

    target = None
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        if kind == "wrist":
            target = _compute_wrist_placement(lm, w, h, cfg)
        else:  # eyes / head
            target = compute_face_placement(lm, w, h, cfg)

    print(f"[debug] {kind} detected={target is not None}")

    placement = smooth_point(sid, target)
    if placement is None:
        return frame, False

    cx, cy, target_w, angle = placement
    target_w = int(max(24, min(target_w, w * 0.35)))
    ar = overlay_img.shape[0] / float(overlay_img.shape[1])
    target_h = int(max(24, target_w * ar))

    resized = smart_resize(overlay_img, target_w, target_h)
    rotated = rotate_rgba(resized, -angle)
    frame = alpha_paste(frame, rotated, cx, cy)
    return frame, target is not None


# ---------------------------------------------------------------- dispatcher


def render_overlay(sid, frame, overlay_img, category):
    """Picks the right rendering path based on the product's configured anchor."""
    if overlay_img is None or overlay_img.size == 0:
        return frame, False

    cfg = PRODUCT_CONFIG.get(category, PRODUCT_CONFIG["apparel"])
    anchor = cfg["anchor"]

    if anchor == "torso":
        return render_torso_overlay(sid, frame, overlay_img, cfg)
    if anchor == "wrist":
        return render_point_overlay(sid, frame, overlay_img, cfg, "wrist")
    # eyes / head
    return render_point_overlay(sid, frame, overlay_img, cfg, "face")


# ---------------------------------------------------------------- events


@sio.event
async def connect(sid, environ):
    print(f"[socket] connected: {sid}")


@sio.event
async def disconnect(sid):
    track_state.pop(sid, None)
    track_state.pop(sid + "_quad", None)
    print(f"[socket] disconnected: {sid}")


@sio.on("update_overlay")
async def handle_update_overlay(sid, data):
    global current_overlay_img, current_product_type
    try:
        if data.get("category"):
            requested = str(data["category"]).lower()
            if requested not in PRODUCT_CONFIG:
                print(f"[warn] Unknown category '{requested}' - falling back to 'apparel'. "
                      f"Valid categories: {list(PRODUCT_CONFIG.keys())}")
            current_product_type = requested
            print(f"[ok] Category: {current_product_type}")

        if data.get("image_data"):
            img = decode_image_bytes(data["image_data"])
            if img is None:
                await sio.emit("overlay_status",
                               {"ok": False, "message": "Failed to decode image"}, room=sid)
                return
            current_overlay_img = img
            track_state.pop(sid, None)
            track_state.pop(sid + "_quad", None)
            await sio.emit("overlay_status",
                           {"ok": True, "message": "Product loaded"}, room=sid)
    except Exception as e:
        print(f"[error] update_overlay: {e}")
        await sio.emit("overlay_status", {"ok": False, "message": str(e)}, room=sid)


@sio.on("stream_frame")
async def handle_stream_frame(sid, data):
    try:
        raw = data.get("image", "")
        if not raw:
            return
        payload = raw.split(",", 1)[1] if "," in raw else raw
        nparr = np.frombuffer(base64.b64decode(payload), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        detected = False
        if current_overlay_img is not None:
            frame, detected = render_overlay(sid, frame, current_overlay_img,
                                             current_product_type)

        out = cv2_to_base64(frame)
        if out:
            await sio.emit("processed_frame",
                           {"image": out,
                            "body_detected": bool(detected),
                            "has_product": current_overlay_img is not None},
                           room=sid)
    except Exception as e:
        print(f"[error] stream_frame: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)


# """
# Virtual Try-On Socket.IO server (Anti-Flicker & Sharp Image Fix)
# """

# import base64
# import io
# import math

# import cv2
# import numpy as np
# import socketio
# import uvicorn
# from PIL import Image

# try:
#     import pillow_avif  # noqa: F401
# except ImportError:
#     print("[warn] pillow-avif-plugin missing")

# import mediapipe as mp

# # ---------------------------------------------------------------- config

# PRODUCT_CONFIG = {
#     "apparel": {"width_mult": 1.95, "y_offset": 0.34, "anchor": "torso"},
#     "tshirt":  {"width_mult": 1.95, "y_offset": 0.34, "anchor": "torso"},
#     "jacket":  {"width_mult": 2.15, "y_offset": 0.36, "anchor": "torso"},
#     "eyewear": {"width_mult": 2.10, "y_offset": 0.00, "anchor": "eyes"},
#     "cap":     {"width_mult": 2.40, "y_offset": -0.55, "anchor": "head"},
#     "watch":   {"width_mult": 0.62, "y_offset": 0.0, "anchor": "wrist"},
# }

# # --- Stabilization Tuning ---
# SMOOTHING = 0.82          # 0.82 rakha gaya hai taake jhatke na lagein
# HOLD_FRAMES = 45          # Missed frame par shirt turant gayab na ho
# MIN_VISIBILITY = 0.35     
# DEADZONE_PX = 3.0         # 3 pixels se kam movement par shirt hilna band kar degi (Freeze)
# DEADZONE_ANGLE = 1.5      # 1.5 degree se kam rotation tilt ko ignore karega (Anti-Blink)
# JPEG_QUALITY = 90         # Quality increase kar di hai taake dhundla na dikhe

# # ---------------------------------------------------------------- state

# current_overlay_img = None      
# current_product_type = "apparel"

# track_state = {}

# mp_pose = mp.solutions.pose
# PL = mp_pose.PoseLandmark

# # Static image mode = True stabilization ke liye best hai SocketIO streams mein
# pose_detector = mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=1,
#     enable_segmentation=False,
#     min_detection_confidence=0.5,
# )

# sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
# app = socketio.ASGIApp(sio)

# # ---------------------------------------------------------------- helpers

# def trim_alpha(img):
#     if img is None or img.shape[2] < 4:
#         return img
#     alpha = img[:, :, 3]
#     ys, xs = np.where(alpha > 8)
#     if ys.size == 0 or xs.size == 0:
#         return img
#     return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# def decode_image_bytes(b64_string):
#     try:
#         encoded = b64_string.split(",", 1)[1] if "," in b64_string else b64_string
#         pil_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
#         bgra = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
#         bgra = trim_alpha(bgra)
#         print(f"[ok] Overlay loaded: {bgra.shape[1]}x{bgra.shape[0]}")
#         return bgra
#     except Exception as e:
#         print(f"[error] decode_image_bytes: {e}")
#         return None


# def cv2_to_base64(img):
#     ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
#     if not ok:
#         return None
#     return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


# def rotate_rgba(img, angle_deg):
#     if abs(angle_deg) < 0.5:
#         return img
#     h, w = img.shape[:2]
#     diag = int(math.hypot(w, h)) + 2
#     M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
#     M[0, 2] += (diag - w) / 2.0
#     M[1, 2] += (diag - h) / 2.0
#     return cv2.warpAffine(
#         img, M, (diag, diag),
#         flags=cv2.INTER_LANCZOS4, # High quality sharp rotation
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=(0, 0, 0, 0),
#     )


# def alpha_paste(frame, overlay, cx, cy):
#     fh, fw = frame.shape[:2]
#     oh, ow = overlay.shape[:2]

#     ox, oy = int(round(cx - ow / 2.0)), int(round(cy - oh / 2.0))

#     x1, y1 = max(0, ox), max(0, oy)
#     x2, y2 = min(fw, ox + ow), min(fh, oy + oh)
#     if x2 <= x1 or y2 <= y1:
#         return frame

#     sx1, sy1 = x1 - ox, y1 - oy
#     crop = overlay[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]

#     if crop.shape[2] == 4:
#         alpha = (crop[:, :, 3:4].astype(np.float32)) / 255.0
#         roi = frame[y1:y2, x1:x2].astype(np.float32)
#         blended = alpha * crop[:, :, :3].astype(np.float32) + (1.0 - alpha) * roi
#         frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
#     else:
#         frame[y1:y2, x1:x2] = crop[:, :, :3]
        
#     return frame


# def _pt(lm, idx, w, h):
#     p = lm[idx]
#     return np.array([p.x * w, p.y * h], dtype=np.float32), p.visibility


# def compute_placement(landmarks, w, h, category):
#     cfg = PRODUCT_CONFIG.get(category, PRODUCT_CONFIG["apparel"])
    
#     ls, v1 = _pt(landmarks, PL.LEFT_SHOULDER, w, h)
#     rs, v2 = _pt(landmarks, PL.RIGHT_SHOULDER, w, h)
#     if min(v1, v2) < MIN_VISIBILITY:
#         return None

#     lh, v3 = _pt(landmarks, PL.LEFT_HIP, w, h)
#     rh, v4 = _pt(landmarks, PL.RIGHT_HIP, w, h)

#     shoulder_mid = (ls + rs) / 2.0
#     shoulder_w = float(np.linalg.norm(ls - rs))
#     if shoulder_w < 12:
#         return None

#     if min(v3, v4) >= MIN_VISIBILITY:
#         hip_mid = (lh + rh) / 2.0
#         torso_len = float(np.linalg.norm(hip_mid - shoulder_mid))
#         direction = (hip_mid - shoulder_mid) / max(torso_len, 1e-5)
#     else:
#         torso_len = shoulder_w * 1.5
#         direction = np.array([0.0, 1.0], dtype=np.float32)

#     center = shoulder_mid + direction * (torso_len * cfg["y_offset"])
#     dx, dy = (rs - ls)
#     angle = math.degrees(math.atan2(dy, dx))
#     if angle > 90:
#         angle -= 180
#     elif angle < -90:
#         angle += 180
        
#     return float(center[0]), float(center[1]), shoulder_w * cfg["width_mult"], angle


# def smooth(sid, target):
#     """Smart Deadzone EMA Smoothing to stop shirt blinking & flickering."""
#     st = track_state.get(sid)

#     if target is None:
#         if st is None:
#             return None
#         st["miss"] += 1
#         if st["miss"] > HOLD_FRAMES:
#             track_state.pop(sid, None)
#             return None
#         return st["cx"], st["cy"], st["w"], st["angle"]

#     cx, cy, width, angle = target
#     if st is None:
#         track_state[sid] = {"cx": cx, "cy": cy, "w": width, "angle": angle, "miss": 0}
#         return cx, cy, width, angle

#     # --- Deadzone calculation (Shirt hilne se rokne ke liye) ---
#     dx = abs(cx - st["cx"])
#     dy = abs(cy - st["cy"])
#     d_ang = abs(((angle - st["angle"] + 180) % 360) - 180)

#     # Agar choti movement hai toh shirt ko jagah par freeze rakho
#     if dx < DEADZONE_PX and dy < DEADZONE_PX and d_ang < DEADZONE_ANGLE:
#         st["miss"] = 0
#         return st["cx"], st["cy"], st["w"], st["angle"]

#     a = SMOOTHING
#     st["cx"] = a * st["cx"] + (1 - a) * cx
#     st["cy"] = a * st["cy"] + (1 - a) * cy
#     st["w"] = a * st["w"] + (1 - a) * width
    
#     diff_ang = ((angle - st["angle"] + 180) % 360) - 180
#     st["angle"] = st["angle"] + (1 - a) * diff_ang
#     st["miss"] = 0
    
#     return st["cx"], st["cy"], st["w"], st["angle"]


# def render_overlay(sid, frame, overlay_img, category):
#     if overlay_img is None or overlay_img.size == 0:
#         return frame, False

#     h, w = frame.shape[:2]

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     rgb.flags.writeable = False
#     result = pose_detector.process(rgb)

#     target = None
#     if result.pose_landmarks:
#         target = compute_placement(result.pose_landmarks.landmark, w, h, category)

#     placement = smooth(sid, target)
#     if placement is None:
#         return frame, False

#     cx, cy, target_w, angle = placement

#     target_w = int(max(24, min(target_w, w * 3)))
#     ar = overlay_img.shape[0] / float(overlay_img.shape[1])
#     target_h = int(max(24, target_w * ar))

#     # INTER_LANCZOS4 image ko HD aur sharp rakhta hai
#     resized = cv2.resize(overlay_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
#     rotated = rotate_rgba(resized, -angle)

#     frame = alpha_paste(frame, rotated, cx, cy)
#     return frame, target is not None

# # ---------------------------------------------------------------- events

# @sio.event
# async def connect(sid, environ):
#     print(f"[socket] connected: {sid}")

# @sio.event
# async def disconnect(sid):
#     track_state.pop(sid, None)
#     print(f"[socket] disconnected: {sid}")

# @sio.on("update_overlay")
# async def handle_update_overlay(sid, data):
#     global current_overlay_img, current_product_type
#     try:
#         if data.get("category"):
#             current_product_type = str(data["category"]).lower()

#         if data.get("image_data"):
#             img = decode_image_bytes(data["image_data"])
#             if img is not None:
#                 current_overlay_img = img
#                 track_state.pop(sid, None)
#                 await sio.emit("overlay_status", {"ok": True, "message": "Product loaded"}, room=sid)
#     except Exception as e:
#         print(f"[error] update_overlay: {e}")

# @sio.on("stream_frame")
# async def handle_stream_frame(sid, data):
#     try:
#         raw = data.get("image", "")
#         if not raw:
#             return
#         payload = raw.split(",", 1)[1] if "," in raw else raw
#         nparr = np.frombuffer(base64.b64decode(payload), np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if frame is None:
#             return

#         detected = False
#         if current_overlay_img is not None:
#             frame, detected = render_overlay(sid, frame, current_overlay_img, current_product_type)

#         out = cv2_to_base64(frame)
#         if out:
#             await sio.emit("processed_frame",
#                            {"image": out,
#                             "body_detected": bool(detected),
#                             "has_product": current_overlay_img is not None},
#                            room=sid)
#     except Exception as e:
#         print(f"[error] stream_frame: {e}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)