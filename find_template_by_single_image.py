import cv2
import numpy as np

def find_template(img, template_path, threshold=0.8, label="Detected"):
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    h, w = template.shape[:2]

    detected = False  # A flag to check if the item has been detected and labeled

    # Check for the template at different scales
    for scale in np.linspace(0.8, 1.4, 20):  # Change the range and step size as needed
        if detected:  # If already detected, break out of the loop
            break

        new_w, new_h = int(w * scale), int(h * scale)
    
        if new_w < 1 or new_h < 1:  # Check to ensure scaled width and height are greater than 0
            continue

        resized_template = cv2.resize(template, (new_w, new_h))
        h, w = resized_template.shape[:2]
        
        if h > img.shape[0] or w > img.shape[1]:  # if the resized template is bigger than the image, skip
            continue

        res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):  # Loop through detected locations
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            cv2.putText(img, label, (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            detected = True
            break  # Break after first detection

    return img

# Start with the base image
img_base = cv2.imread("gags 2 Screenshot 2023-10-14 224714.png", cv2.IMREAD_COLOR)

# Example usage:
img_result = find_template(img_base, "images/friends_btn.png", threshold=0.55, label="Friends Button")
img_result = find_template(img_base, "images/sticker_book.png", threshold=0.55, label="Sticker Book")
img_result = find_template(img_base, "images/cupcake.png", threshold=0.65, label="Cupcake")
img_result = find_template(img_base, "images/squirting-flower.png", threshold=0.65, label="Squirting Flower")

cv2.imshow("Detected Elements", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()