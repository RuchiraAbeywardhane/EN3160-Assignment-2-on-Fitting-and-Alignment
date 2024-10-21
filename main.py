import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = 'the_berry_farms_sunflower_field.jpeg'
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

# Function to apply Laplacian of Gaussian
def laplacian_of_gaussian(image, sigma):
    blurred = cv.GaussianBlur(image, (0, 0), sigma)  # Apply Gaussian blur
    log = cv.Laplacian(blurred, cv.CV_64F)           # Apply Laplacian
    return log

# Detect extrema points in scale-space
def detect_blobs(image, sigma_values):
    blobs = []  # Store blobs

    for sigma in sigma_values:
        log_image = laplacian_of_gaussian(image, sigma)
        threshold = np.percentile(np.abs(log_image), 99.7)
        log_extrema = (np.abs(log_image) > threshold)
        labeled_blobs, _ = cv.connectedComponents(np.uint8(log_extrema))
        
        for label in range(1, np.max(labeled_blobs) + 1):
            blob_pixels = np.where(labeled_blobs == label)
            if len(blob_pixels[0]) > 0:
                y_center = np.mean(blob_pixels[0])
                x_center = np.mean(blob_pixels[1])
                radius = sigma * np.sqrt(2)
                blobs.append({'center': (int(x_center), int(y_center)), 'radius': int(radius)})
    
    return blobs

# Function to draw circles on the image
def draw_circles(image, blobs):
    img_with_circles = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    for blob in blobs:
        cv.circle(img_with_circles, blob['center'], blob['radius'], (0, 255, 0), 2)
    
    return img_with_circles

# Define range of sigma values (scales)
sigma_values = np.arange(2, 20, 2)

# Detect blobs in the image
blobs = detect_blobs(img, sigma_values)

# Sort blobs by radius to find the largest circles
blobs_sorted = sorted(blobs, key=lambda x: x['radius'], reverse=True)
largest_blobs = blobs_sorted[:5]

# Draw the circles on the image
img_with_circles = draw_circles(img, largest_blobs)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(img_with_circles[..., ::-1])
plt.title('Detected Circles in Sunflower Field')
plt.show()

# Report largest circles' parameters
print("Largest Circles' Parameters:")
for i, blob in enumerate(largest_blobs):
    print(f"Circle {i+1}: Center = {blob['center']}, Radius = {blob['radius']}")

# Report sigma range
print(f"Sigma values used: {sigma_values}")

