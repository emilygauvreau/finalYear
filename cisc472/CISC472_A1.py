'''
Assignment 1 - CISC 472
Written by Emily Gauvreau
17emg - 20074874
'''
import cv2
import numpy

# Question 1 - Read Image
# Takes the filepath as a parameter and returns the image as a numpy array
# Sourced from walkthrough as mentioned in the Q&A sessions
def readImage(filePath):
    image = cv2.imread(filePath)
    greyscale = True
    row = 0
    while row < image.shape[0] and greyscale:
        column = 0
        while column < image.shape[1] and greyscale:
            pixel = image[row][column]
            if not numpy.all(pixel == pixel[0]):
                greyscale = False
            column += 1
        row += 1
    if greyscale:
        image = image[:,:,0]
    return image


# Question 2 - Display Image
# Takes the image as an array as a parameter in addition to a title for the window
# Does not return anything - simply displays the window and waits for a key to continue the program
def displayImage(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)


# Question 3 - Linear Filters
# Takes a square kernel of any size and an image as parameters, returns the image resulting from the kernel being applied
# Creates an empty array of zeros such that as the filter is applied to each window and the result can be updated in the new array
# Uses the numpy.sum() function to sum all of the values in the array produced from multiplying the filter by the window
def applyLinearFilter(squareKernel, image):

    filteredImage = numpy.zeros(image.shape)
    padding = squareKernel.shape[0] // 2 # return number of padding needed around the original image
    padImage = numpy.pad(image, padding, mode="edge")

    for row in range(padding, image.shape[0]):
        for column in range(padding, image.shape[1]):
            window = padImage[(row - padding):(row + padding + 1), (column - padding):(column + padding + 1)]
            intensity = int(round(numpy.sum(squareKernel * window)))
            filteredImage[row][column] = intensity
    filteredImage = filteredImage.astype("uint8")
    return filteredImage


# Test Function for Question 3 
# Using the kernels supplied in the lecture notes, each one is applied to the original image
# They are then displayed in a new window. A filename is required as a parameter but no return value is produced.
def testLinearFilters(fileName):
    originalImage = readImage(fileName)
    displayImage(originalImage, "Original Greyscale Image")

    smoothingFilter = (1/10.0) * numpy.array([[1,1,1],[1,2,1],[1,1,1]])
    applySmooth = applyLinearFilter(smoothingFilter, originalImage)
    displayImage(applySmooth, "Smoothing Filter")

    edgeFilter = numpy.array([[0,0,0],[0,-1,1],[0,0,0]])
    applyEdge = applyLinearFilter(edgeFilter, originalImage)
    displayImage(applyEdge, "Edge Detection")

    sharpeningFilter = numpy.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    applySharpen = applyLinearFilter(sharpeningFilter, originalImage)
    displayImage(applySharpen, "Sharpen Filter")


# Question 4 - Non-Linear Filter
# Takes an image and the size of the image filter, returns the image after the filter has been applied
# Using similar parsing logic to the linear filters a specific window of the array is specified and the numpy.mean() function
# is used to calculate the mean of that window and save it within the newly created array of zeros
def applyMeanFilter(image, filterSize):

    filteredImage = numpy.zeros(image.shape)
    padding = filterSize // 2 # return needed edge
    padImage = numpy.pad(image, padding) # pad with zeros so that the extra values don't contribute to mean

    for row in range(padding, image.shape[0]):
        for column in range(padding, image.shape[1]):
            window = padImage[(row - padding):(row + padding + 1),(column - padding):(column + padding + 1)]
            meanVal = int(round(numpy.mean(window)))
            filteredImage[row][column] = meanVal
    filteredImage = filteredImage.astype("uint8")
    return filteredImage


# Test Function for Question 4
# The function accepts a fileName as a parameter to be used as the original reference and no return value is produced
# The tests utilize the mean filtering function with an array of 3x3 and 7x7 to visualize the difference
def testMeanFilter(fileName):
    
    originalImage = readImage(fileName)
    displayImage(originalImage, "Original Greyscale Image")

    # Size 3 Matrix
    applyMean3 = applyMeanFilter(originalImage, 3)
    displayImage(applyMean3, "Mean Filter with 3x3")

    # Size 7 Matrix
    applyMean7 = applyMeanFilter(originalImage, 7)
    displayImage(applyMean7, "Mean Filter with 7x7")


# Question 5 - Depth Modification
# Function accepts the image to be modified and the desired depth it will become. It returns the updated image.
# It originally checks what bit depth the image currently is and applies an equation to either decrease or 
# increase the depth of each pixel 
def depthMod(image, depth):
    ## I am making the assumption that the depth is provided as bits NOT bytes
    if image.dtype == numpy.uint8:
        divisor = 255
    elif image.dtype == numpy.uint16:
        divisor = 65535
    elif image.dtype == numpy.uint32:
        divisor = 4,294,967,295
    
    depthBytes = 2 ** depth
    maxVal = max(range(depthBytes))
    modImage = numpy.zeros(image.shape)

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            pixel = image[row][column]
            modImage[row][column] = int(round(pixel * maxVal / divisor))

    modImage = modImage.astype("uint16")
    return modImage

# Test Function for Question 5
# Accepts the filename for the image to be modified as a parameter and has no return value
# Using the function in question 5 it displays both the original image and the image after alteration
def testDepthMod(fileName):
    originalImage = readImage(fileName)
    displayImage(originalImage, "Original Greyscale Image")

    applyDepth = depthMod(originalImage, 12)
    displayImage(applyDepth, "Image Represented as CT 12-bit Depth")


# Question 6 - Contrast Enhancement
# Accepts an image as well as an upper and lower boundary for the range to be enhanced. It returns the image after modification.
# The function creates an empty array of zeros and then iterates over every pixel, altering the pixel based on the equation provided in the assignment
# Once the result is calculated it is added to the pixel position of the new array.
def contrastEnhancement(image, L1, L2):
    ## The question did not cover the situation where f(m,n) = L2 so I assumed it belonged in the middle range as per lecture notes

    depth = 255 # As the images being studied are one channel and 8-bit, a max depth of 255 is used.
    modImage = numpy.zeros(image.shape)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            pixel = image[row][column]
            if pixel < L1:
                modImage[row][column] = 0
            elif pixel >= L1 and pixel <= L2:
                modImage[row][column] = depth * ((pixel - L1)/L2-L1)
            else: # pixel > L2
                modImage[row][column] = depth
    modImage = modImage.astype("uint8")
    return modImage


# Test Function for Question 6
# Takes a filename for the original image to be altered as a parameter and does not have any returns. 
# Utilizes the above function 4 times each with a different L1 and L2 value to enhance different regions
def testContrast(filename):

    originalImage = readImage(filename)
    displayImage(originalImage, "Original Greyscale Image")

    range1 = contrastEnhancement(originalImage, 0.25*255, 0.75*255)
    displayImage(range1, "Contrast of Range 1")

    range2 = contrastEnhancement(originalImage, 0.45*255, 0.55*255)
    displayImage(range2, "Contrast of Range 2")
    
    range3 = contrastEnhancement(originalImage, 0.10*255, 0.60*255)
    displayImage(range3, "Contrast of Range 3")
    
    range4 = contrastEnhancement(originalImage, 0.40*255, 0.90*255)
    displayImage(range4, "Contrast of Range 4")


# Helper Function for Question 7 to generate a random ellipse
# Takes an image where the ellipse should be added as a parameter and returns the image after application
# Uses a series of numpy functions to calculate the arc of the ellipse
# The built in functions are described within the code for ease of reading
def generateEllipse(image):

    color = 255 # white
    imageHeight = len(image)
    imageWidth = len(image[0])
    
    # randomizes how tall and wide the ellipse will be (uses 5 and 3 to ensure that the ellipse isn't too large)
    axisW = numpy.random.randint(imageWidth / 5, imageWidth / 3)
    axisH = numpy.random.randint(imageHeight / 5, imageHeight / 3)

    # ensures that the center is always within the boundary of the image
    yCenter = numpy.random.randint(0, imageHeight)
    xCenter = numpy.random.randint(0, imageWidth)

    # Evenly space 360 * 2 points across 360 space creating the outline of the ellipse
    angles = numpy.linspace(0, 360, 360 * 2)
    # Iterate over angles to determine x and y position and range to fill
    for i in angles:
        currentAngle = i * numpy.pi/180
        yPos = xCenter + axisW * numpy.sin(currentAngle)
        xPos = yCenter + axisH * numpy.cos(currentAngle)
        
        # Clip x and y position to boundaries of image so it doesn't go over
        xPos = numpy.clip(xPos, 0, imageWidth)
        yPos = numpy.clip(yPos, 0, imageHeight)

        # Convert max positions to integers for slicing
        row, column = int(xPos), int(yPos)
        
        if i > 270:
            image[yCenter:row, column:xCenter] = color
        elif i > 180:
            image[row:yCenter, column:xCenter] = color
        elif i > 90:
            image[row:yCenter, xCenter:column] = color
        elif i > 0:
            image[yCenter:row, xCenter:column] = color

    # edges of ellipse can be cut up but center must be in image
    return image


# Question 7 - Generate a Simulated Image
# Takes the height, width as the dimensions of the image to be created. Returns the created image as array
# Generates a plain black background and passes it to the helper function generateEllipse()
def generateImage(height, width):
    simulatedImage = numpy.zeros((height, width), numpy.uint8)
    modifiedImage = generateEllipse(simulatedImage)
    return modifiedImage


# Test Function for Question 7
# It takes the dimensions of the image and applies 3 different mean filters and displays them
def testMean1(height, width):
    simulatedImage = generateImage(height, width)
    size3 = applyMeanFilter(simulatedImage, 3)
    displayImage(size3, "Mean Filter 3x3")

    size7 = applyMeanFilter(simulatedImage, 7)
    displayImage(size7, "Mean Filter 7x7")

    size15 = applyMeanFilter(simulatedImage, 15)
    displayImage(size15, "Mean Filter 15x15")


# Test Function for Question 7 
# It takes the dimensions of the image and applies the same filter a different number of times  and displays them
def testMean2(height, width):
    simulatedImage = generateImage(height, width)
    
    size5 = applyMeanFilter(simulatedImage, 5)
    displayImage(size5, "Mean Filter 5x5 Once")

    for i in range(10):
        tenTimes = applyMeanFilter(simulatedImage, 5)
    displayImage(tenTimes, "Mean Filter 5x5 Ten Times")

    for i in range(100):
        oneHunTimes = applyMeanFilter(simulatedImage, 5)
    displayImage(oneHunTimes, "Mean Filter 5x5 One Hundred Times")


def main():
    
    testLinearFilters("greyscale.jpg")
    testMeanFilter("greyscale.jpg")
    testDepthMod("greyscale.jpg")
    testContrast("greyscale.jpg")
    testMean1(512, 512)
    testMean2(512, 512)

if __name__ == "__main__":
    main()
