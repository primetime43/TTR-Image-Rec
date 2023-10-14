import cv2
import numpy as np

def find_template(img, template_path, threshold=0.8, label="Detected"):
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    h, w = template.shape[:2]

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    cv2.putText(img, label, (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    return img

# Start with the base image
img_base = cv2.imread("ttr Screenshot 2023-10-13 143250.png", cv2.IMREAD_COLOR)

# Example usage:
img_result = find_template(img_base, "images/friends_btn.png", threshold=0.65, label="Friends Button")
img_result = find_template(img_base, "images/sticker_book.png", threshold=0.55, label="Sticker Book")

cv2.imshow("Detected Elements", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()