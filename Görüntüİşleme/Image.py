import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Oluşacaklar için klasör
os.makedirs("output", exist_ok=True)

# 1. Histogram
def histogram(imagePath, outputFolder):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    histogram, binEdges = np.histogram(image, bins=256, range=(0, 255))
    
    plt.figure(figsize=(8, 6))
    plt.bar(binEdges[:-1], histogram, width=1, color='gray', edgecolor='black')
    plt.title(f'Histogram of - {os.path.basename(imagePath)}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    outputFile = os.path.join(outputFolder, f"histogram_{os.path.basename(imagePath).split('.')[0]}.png")
    plt.savefig(outputFile)
    plt.close()

# 2. Binarization
def binarization(imagePath, threshold, outputFolder):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    _, binaryImage = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    outputFile = os.path.join(outputFolder, f"binary_{os.path.basename(imagePath).split('.')[0]}.png")
    cv2.imwrite(outputFile, binaryImage)

# 3. Resmi birden çok Bölgelere Ayırma
def segment(imagePath, thresholds, outputFolder):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    regions = np.zeros_like(image)
    # Eşik değerlere göre
    for i, (low, high) in enumerate(zip([0] + thresholds, thresholds + [255])):
        regions[(image >= low) & (image < high)] = int((i + 1) * (255 / (len(thresholds) + 1)))
    
    outputFile = os.path.join(outputFolder, f"segmented_{os.path.basename(imagePath).split('.')[0]}.png")
    cv2.imwrite(outputFile, regions)

    return regions

# 4. Morfolojik İşlemler
def morpho(imagePath, outputFolder):
    binaryImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    ker = np.ones((3, 3), np.uint8)

    # Erozyon işlemi
    erosion = cv2.erode(binaryImage, ker, iterations=1)
    cv2.imwrite(os.path.join(outputFolder, f"erosion_{os.path.basename(imagePath)}"), erosion)
    
    # Dilatasyon işlemi
    dilation = cv2.dilate(binaryImage, ker, iterations=1)
    cv2.imwrite(os.path.join(outputFolder, f"dilation_{os.path.basename(imagePath)}"), dilation)
    
    # Açma işlemi
    opening = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, ker)
    cv2.imwrite(os.path.join(outputFolder, f"opening_{os.path.basename(imagePath)}"), opening)
    
    # Kapama işlemi
    closing = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, ker)
    cv2.imwrite(os.path.join(outputFolder, f"closing_{os.path.basename(imagePath)}"), closing)

# 5. Region Growing
def regionGrowing(imagePath, seedPoints, outputFolder):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    rows, cols = image.shape
    visit = np.zeros((rows, cols), dtype=bool)
    outputImage = np.zeros_like(image)
    direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Her seed için 
    for seed_point in seedPoints:
        stack = [seed_point]
        while stack:
            x, y = stack.pop()
            if visit[x, y]:
                continue
            visit[x, y] = True
            outputImage[x, y] = 255
            for dx, dy in direction:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visit[nx, ny]:
                    if abs(int(image[nx, ny]) - int(image[x, y])) < 7:  # Fark eşiği
                        stack.append((nx, ny))
    
    outputFile = os.path.join(outputFolder, f"regionGrowing_{os.path.basename(imagePath).split('.')[0]}.png")
    cv2.imwrite(outputFile, outputImage)

# 6. Histogram Karşılaştırma
def histogramEqualization(imagePath, outputFolder):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(image)
    
    # Orijinal ve equalization histogramları karşılaştırma
    originalHistogram, original_bins = np.histogram(image, bins=256, range=(0, 255))
    equalizedHistogram, equalized_bins = np.histogram(equalized_image, bins=256, range=(0, 255))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(original_bins[:-1], originalHistogram, width=1, color='gray')
    plt.title("Original Histogram")
    
    plt.subplot(1, 2, 2)
    plt.bar(equalized_bins[:-1], equalizedHistogram, width=1, color='blue')
    plt.title("Equalized Histogram")
    plt.savefig(os.path.join(outputFolder, f"histogram_comparison_{os.path.basename(imagePath)}"))
    plt.close()
    
    cv2.imwrite(os.path.join(outputFolder, f"equalized_{os.path.basename(imagePath)}"), equalized_image)

# resimler
images = ["hw1_images/Fig0107(a)(chest-xray-vandy).tif", 
          "hw1_images/Fig0120(a)(ultrasound-fetus1).tif", 
          "hw1_images/Fig0304(a)(breast_digital_Xray).tif", 
          "hw1_images/Fig0359(a)(headCT_Vandy).tif"]

thresholdsSegmentation = [[55, 105, 155], [65, 125, 185], [40, 80, 120, 160], [30, 90, 150]]
binaryThresholds = [105, 125, 145, 165]

for img, thresholds in zip(images, thresholdsSegmentation):
    histogram(img, "output")
    binarization(img, binaryThresholds[images.index(img)], "output")
    segmented_image = segment(img, thresholds, "output")
    morpho(img, "output")
    histogramEqualization(img, "output")

    seedPoints = []
    for region in np.unique(segmented_image):
        if region != 0: 
            region_pixels = np.where(segmented_image == region)
            centerx = int(np.mean(region_pixels[0]))
            centery = int(np.mean(region_pixels[1]))
            seedPoints.append((centerx, centery))
    regionGrowing(img, seedPoints, "output")