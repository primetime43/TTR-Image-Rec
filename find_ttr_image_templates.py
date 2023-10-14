import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time

def non_max_suppression(boxes, overlapThresh):
    print('In non_max_suppression')
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]
            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick].astype("int")

def find_template(img, template_path, threshold=0.8, label="Detected"):
    print('In find template')
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    h, w = template.shape[:2]

    # Check if template size is smaller or equal to the source image size
    if h > img.shape[0] or w > img.shape[1]:
        print(f"Template {template_path} is larger than the source image. Skipping this template.")
        return img

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    boxes = []
    for pt in zip(*loc[::-1]):
        boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])

    pick = non_max_suppression(np.array(boxes), 0.3)  # 0.3 is the overlap threshold

    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(img, (startX, startY), (endX, endY), (0,0,255), 2)
        cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    return img

def capture_window(title):
    print('In capture window')
    try:
        win = gw.getWindowsWithTitle(title)[0]
        if win != None:
            x, y, width, height = win.left, win.top, win.width, win.height
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            return np.array(screenshot)
        else:
            return None
    except:
        return None

# Main loop
cv2.namedWindow("Detected Elements", cv2.WINDOW_NORMAL)  # Create a named window
cv2.resizeWindow("Detected Elements", 800, 600)  # Set initial window size

while True:
    img_base = capture_window("Toontown Rewritten")
    print('In main loop')
    
    if img_base is not None and img_base.shape[0] > 0 and img_base.shape[1] > 0:
        # Convert the screenshot from RGB to BGR format (for OpenCV)
        img_base = cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR)
        
        # Find templates
        img_result = find_template(img_base, "images/friends_btn.png", threshold=0.65, label="Friends Button")
        img_result = find_template(img_base, "images/sticker_book.png", threshold=0.55, label="Sticker Book")

        # Resize the result for display
        height, width = img_result.shape[:2]
        img_resized = cv2.resize(img_result, (int(width * 0.5), int(height * 0.5)))  # 50% of original size

        # Show result
        cv2.imshow("Detected Elements", img_resized)
        
        key = cv2.waitKey(10000)  # Wait for 10 seconds or until a key is pressed
        if key & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()