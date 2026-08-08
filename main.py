import cv2
import mediapipe as mp
import pyautogui
import time
import logging
from collections import deque

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    filename="gesture_log.txt",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
SWIPE_THRESHOLD_RATIO = 0.4
SWIPE_HISTORY_LEN     = 6
SWIPE_MIN_FRAMES      = 3
COOLDOWN              = 1.0
FIST_HOLD_FRAMES      = 8
PEACE_HOLD_FRAMES     = 8
FEEDBACK_DURATION     = 1.2
CAM_WIDTH             = 640
CAM_HEIGHT            = 480
MAX_CAM_FAILURES      = 30

# =========================
# WEBCAM SETUP
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Falling back to keyboard-only mode.")
    log.error("Webcam failed to open.")
    cap = None

# =========================
# MEDIAPIPE HANDS
# =========================
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# =========================
# STATE VARIABLES
# =========================
x_history          = deque(maxlen=SWIPE_HISTORY_LEN)
last_action_time   = 0
paused             = False
fist_frame_count   = 0
peace_frame_count  = 0
cam_fail_count     = 0
blank_screen_active = False

# UI feedback
feedback_text  = ""
feedback_color = (255, 255, 255)
feedback_expiry = 0

# Blank screen window name
BLANK_WIN = "Blank Screen  [peace sign to close]"

# =========================
# HELPERS
# =========================
def is_fist(landmarks):
    thumb_closed   = landmarks[4][0] < landmarks[3][0]
    finger_tips    = [8, 12, 16, 20]
    fingers_closed = all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in finger_tips)
    return thumb_closed and fingers_closed


def is_peace(landmarks):
    index_up   = landmarks[8][1]  < landmarks[6][1]
    middle_up  = landmarks[12][1] < landmarks[10][1]
    ring_down  = landmarks[16][1] > landmarks[14][1]
    pinky_down = landmarks[20][1] > landmarks[18][1]
    thumb_down = landmarks[4][0]  < landmarks[3][0]
    return index_up and middle_up and ring_down and pinky_down and thumb_down


def hand_size(landmarks):
    x0, y0 = landmarks[0]
    x9, y9 = landmarks[9]
    return ((x9 - x0) ** 2 + (y9 - y0) ** 2) ** 0.5


def detect_swipe(x_history, dynamic_threshold):
    if len(x_history) < SWIPE_MIN_FRAMES:
        return None
    recent      = list(x_history)[-SWIPE_MIN_FRAMES:]
    total_delta = recent[-1] - recent[0]
    direction   = 1 if total_delta > 0 else -1
    consistent  = all(
        (recent[i] - recent[i - 1]) * direction >= 0
        for i in range(1, len(recent))
    )
    if not consistent:
        return None
    if total_delta >  dynamic_threshold:
        return "right"
    if total_delta < -dynamic_threshold:
        return "left"
    return None


def send_key(key, label, color):
    global feedback_text, feedback_color, feedback_expiry, last_action_time
    try:
        pyautogui.press(key)
    except Exception as e:
        log.warning(f"pyautogui failed: {e}")
    feedback_text    = label
    feedback_color   = color
    feedback_expiry  = time.time() + FEEDBACK_DURATION
    last_action_time = time.time()
    log.info(f"Action: {label}")


def open_blank():
    global blank_screen_active
    blank = __import__('numpy').zeros(
        (pyautogui.size()[1], pyautogui.size()[0], 3),
        dtype=__import__('numpy').uint8
    )
    cv2.namedWindow(BLANK_WIN, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(BLANK_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(BLANK_WIN, blank)
    blank_screen_active = True
    log.info("Blank screen opened.")


def close_blank():
    global blank_screen_active
    cv2.destroyWindow(BLANK_WIN)
    blank_screen_active = False
    log.info("Blank screen closed.")


def draw_ui(img, paused, blank_screen_active, feedback_text, feedback_color, feedback_expiry):
    current_time = time.time()
    h, w = img.shape[:2]

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    if blank_screen_active:
        cv2.putText(img, "BLANK SCREEN  [peace sign to close]",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    elif paused:
        cv2.putText(img, "PAUSED  [fist to resume]",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    else:
        if current_time < feedback_expiry:
            cv2.putText(img, feedback_text, (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, feedback_color, 2)
        else:
            cv2.putText(img, "ACTIVE  |  swipe:slide  fist:pause  peace:blank",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.putText(img, "Q: quit  |  SPACE: pause toggle",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)


# =========================
# KEYBOARD FALLBACK
# =========================
def keyboard_only_mode():
    print("Keyboard-only mode. r=right, l=left, p=pause, q=quit")
    log.info("Keyboard-only mode active.")
    paused = False
    while True:
        key = input("Command: ").strip().lower()
        if key == 'r':
            pyautogui.press("right"); print("→ NEXT SLIDE")
        elif key == 'l':
            pyautogui.press("left");  print("← PREVIOUS SLIDE")
        elif key == 'p':
            paused = not paused;      print("PAUSED" if paused else "ACTIVE")
        elif key == 'q':
            break


# =========================
# MAIN LOOP
# =========================
if cap is None:
    keyboard_only_mode()
else:
    log.info("Gesture controller started.")
    try:
        while True:
            success, img = cap.read()
            if not success:
                cam_fail_count += 1
                log.warning(f"Camera read failed ({cam_fail_count}/{MAX_CAM_FAILURES})")
                if cam_fail_count >= MAX_CAM_FAILURES:
                    print("[ERROR] Camera lost. Switching to keyboard-only mode.")
                    log.error("Camera lost after repeated failures.")
                    cap.release()
                    cv2.destroyAllWindows()
                    keyboard_only_mode()
                    break
                continue
            cam_fail_count = 0

            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            current_time = time.time()

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]
                    if len(landmarks) < 21:
                        continue

                    h_size = hand_size(landmarks)

                    # =========================
                    # PEACE SIGN → TOGGLE BLANK SCREEN
                    # =========================
                    if is_peace(landmarks):
                        peace_frame_count += 1
                    else:
                        peace_frame_count = 0

                    if peace_frame_count == PEACE_HOLD_FRAMES and \
                            (current_time - last_action_time > COOLDOWN):
                        if blank_screen_active:
                            close_blank()
                            feedback_text   = "BLANK SCREEN CLOSED"
                            feedback_color  = (0, 200, 255)
                            feedback_expiry = current_time + FEEDBACK_DURATION
                        else:
                            open_blank()
                            feedback_text   = "BLANK SCREEN OPEN"
                            feedback_color  = (0, 200, 255)
                            feedback_expiry = current_time + FEEDBACK_DURATION
                        last_action_time  = current_time
                        peace_frame_count = 0

                    # =========================
                    # FIST → TOGGLE PAUSE (only when blank screen closed)
                    # =========================
                    if not blank_screen_active:
                        if is_fist(landmarks):
                            fist_frame_count += 1
                        else:
                            fist_frame_count = 0

                        if fist_frame_count == FIST_HOLD_FRAMES and \
                                (current_time - last_action_time > COOLDOWN):
                            paused           = not paused
                            x_history.clear()
                            last_action_time = current_time
                            fist_frame_count = 0
                            label = "PAUSED" if paused else "RESUMED"
                            color = (0, 0, 255) if paused else (0, 255, 0)
                            feedback_text   = label
                            feedback_color  = color
                            feedback_expiry = current_time + FEEDBACK_DURATION
                            log.info(f"Pause toggled: {label}")

                    # =========================
                    # SWIPE → SLIDE NAVIGATION
                    # =========================
                    if not paused and not blank_screen_active and not is_fist(landmarks):
                        index_x, index_y = landmarks[8]
                        x_history.append(index_x)

                        cv2.circle(img, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)

                        dynamic_threshold = h_size * SWIPE_THRESHOLD_RATIO

                        if current_time - last_action_time > COOLDOWN:
                            swipe = detect_swipe(x_history, dynamic_threshold)
                            if swipe == "right":
                                send_key("right", ">> NEXT SLIDE", (0, 255, 0))
                                x_history.clear()
                            elif swipe == "left":
                                send_key("left", "<< PREV SLIDE", (0, 165, 255))
                                x_history.clear()
                    else:
                        x_history.clear()

            else:
                x_history.clear()
                fist_frame_count  = 0
                peace_frame_count = 0

            # =========================
            # UI RENDERING
            # =========================
            draw_ui(img, paused, blank_screen_active,
                    feedback_text, feedback_color, feedback_expiry)
            cv2.imshow("Gesture Presentation Controller", img)

            # =========================
            # KEYBOARD SHORTCUTS
            # =========================
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log.info("Quit by user.")
                break
            elif key == ord(' '):
                paused = not paused
                label  = "PAUSED" if paused else "RESUMED"
                feedback_text   = label
                feedback_color  = (0, 0, 255) if paused else (0, 255, 0)
                feedback_expiry = time.time() + FEEDBACK_DURATION
                log.info(f"Manual pause toggle: {label}")
            elif key == 81 or key == ord('l'):
                send_key("left", "<< PREV SLIDE", (0, 165, 255))
            elif key == 83 or key == ord('r'):
                send_key("right", ">> NEXT SLIDE", (0, 255, 0))

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Gesture controller shut down cleanly.")
