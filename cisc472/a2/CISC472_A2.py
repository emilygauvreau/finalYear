'''
Assignment 2 - CISC 472
Written by Emily Gauvreau
17emg - 20074874
'''
import cv2
import matplotlib.pyplot as plt
import numpy


# Question 1 - Generate a 3D volume of ellipse slices
# Utilizing the helper function generateEllipse it takes the height, width and depth (x,y,z) coordinates for a 3D image
# As well as giving the option to decide where the center point is based and returns a 3D volume with a white ellipse on
# black background on every slice
def generate3DEllipse(height, width, depth, fixedCenterX, fixedCenterY):
    volume = numpy.zeros((height, width, depth), numpy.uint8)

    if fixedCenterX is None: 
        xCenter = numpy.random.randint(0, width)
    else:
        xCenter = fixedCenterX
    
    if fixedCenterY is None:
        yCenter = numpy.random.randint(0, height)
    else: 
        yCenter = fixedCenterY
    
    ellipseWidth = numpy.random.randint(width)
    ellipseHeight = numpy.random.randint(height)

    for i in range(depth):
        slice = generateEllipse(ellipseHeight, ellipseWidth, xCenter, yCenter)
        for _ in range(50):
            slice = applyMeanFilter(slice, 3)
        volume[i] = slice

        xCenter = numpy.clip((xCenter + (numpy.random.randint(-10, 10))), 0, 223)
        yCenter = numpy.clip((yCenter + (numpy.random.randint(-10, 10))), 0, 223)
        randomSize = numpy.random.randint(-5, 5)
        multiplier = 1 + (randomSize / 100.0)
        ellipseHeight = round(ellipseHeight * multiplier)
        ellipseWidth = round(ellipseWidth * multiplier)

    return volume

# Question 2 - Split along the axis
# Takes the volume you would like to slice in addition to the slice you would like to be the center
# If no x,y,z is provided it will take the center of the shape
# Returns the visualization of the slices for each anatomical plane
def reslice(volume, x, y, z, pointBased=False):
    if x is None:
        x = volume.shape[0] // 2
    if y is None:
        y = volume.shape[1] // 2
    if z is None:
        z = volume.shape[2] // 2
    imageAxial = volume[x,:,:]
    imageCoronal = volume[:,y,:]
    imageSagittal = volume[:,:,z]
    images = [imageAxial, imageCoronal, imageSagittal]
    
    if pointBased:
        displayMatPlot(images[0], "Center Axial Slice")
        displayMatPlot(images[1], "Center Coronal Slice")
        displayMatPlot(images[2], "Center Sagittal Slice")
    else:
        displayResampledSlices([imageAxial, imageCoronal, imageSagittal], "Center")
    
# Question 3 
# Takes the volume you would like to perform the projection on as a parameter and returns the 3 images produced along each axis
# Using the built in max function along with the axis parameter it creates the slices
def maxIntensity(volume):
    imageAxial = numpy.max(volume, axis=0)
    imageCoronal = numpy.max(volume, axis=1)
    imageSagittal = numpy.max(volume, axis=2)
    return [imageAxial, imageCoronal, imageSagittal]

# Takes the volume you would like to perform the projection on as a parameter and returns the 3 images produced along each axis
# Using the built in sum function along with the axis parameter it creates the slices
def digitalRadiograph(volume):

    imageAxial = numpy.sum(volume, axis=0)
    imageCoronal = numpy.sum(volume, axis=1)
    imageSagittal = numpy.sum(volume, axis=2)
    return [imageAxial, imageCoronal, imageSagittal]


# Question 4
# The function takes the image as well as the upper and lower values of the range and returns a point based binary segementation image
# Using the where function it determines if a pixel is within the range if so it is set to one.
def pointBased(image, lower, upper):
    ## The question did not cover the situation where f(m,n) = range so I assumed it belonged in the middle range 
    modImage = numpy.zeros((image.shape))
    modImage = numpy.where((image >= lower) & (image <= upper), 1, 0)
    if len(image.shape) == 2:
        displayMatPlot(modImage, "Point Based")
    if len(image.shape) == 3:
        x = modImage.shape[0] // 2
        y = modImage.shape[1] // 2
        z = modImage.shape[2] // 2
        reslice(modImage, x, y, z, pointBased=True)
    return modImage

# Question 5
# Takes the image array, the list of seed starting locations and the maximum threshold difference
# Returns the image that is produced when the segmentation occurs. If the difference between two pixels is 
# less than the threshold it is set to 1 otherwise it is set to 0. It has a default parameter that specifies what
# seed region has which label e.g. 1 or 0
def regionGrowing(image, seeds, thresholdDif, labels = {(0,0): 1}):

    binImage = numpy.zeros((image.shape))
    neighbours = [(1,0), (-1,0), (0,-1), (0,1), (1,-1), (1,1), (-1, -1), (-1, 1)]

    currPoint = seeds.pop(0)
    label = labels[currPoint] if currPoint in labels else 1
    antiLabel = 0

    while len(seeds) > 0:

        currValue = image[currPoint[0]][currPoint[1]]
        antiLabel = 0 if label == 1 else 1

        for neighbour in neighbours:
            neighbourX = currPoint[0] + neighbour[0]
            neighbourY = currPoint[1] + neighbour[1]
            
            if not (0 <= neighbourX < image.shape[0] and 0 <= neighbourY < image.shape[1]):
                continue
            if binImage[neighbourX][neighbourY] != label:
                nValue = image[neighbourX][neighbourY]
                if abs(int(currValue) - int(nValue)) <= thresholdDif:
                    binImage[neighbourX][neighbourY] = label
                    nPoint = (neighbourX, neighbourY)
                    
                    if nPoint not in seeds:
                        seeds.append(nPoint)
                        labels[nPoint] = label
                else:
                    binImage[neighbourX][neighbourY] = antiLabel

        currPoint = seeds.pop(0)
        label = labels[currPoint] if currPoint in labels else 1
    
    return binImage



# Helper Function - Display Image with Open CV
# Takes the image as an array as a parameter in addition to a title for the window
# Does not return anything - simply displays the window and waits for a key to continue the program
def displayImage(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)

# Helper Function - Display Image
# Takes the image as an array as a parameter in addition to a title for the window
# Does not return anything - simply displays the window and waits for a key to continue the program
def displayMatPlot(array, windowName="Figure"):
    plt.title(windowName)
    plt.imshow(array, cmap='gray')
    plt.show()

# Helper Functions - Display Images when depth resampling is required
# Takes the images and the title but does not return a value just displays a window
# Calls the helper function resampleDepth() to map the values of the image to 0-255
def displayResampledSlices(images, title):
    for i in range(len(images)):
        images[i] = resampleDepth(images[i])
    displayImage(images[0], title+" Axial Slice")
    displayImage(images[1], title+" Coronal Slice")
    displayImage(images[2], title+" Sagittal Slice")

# Helper Function 
# Takes an array and a filename and saves the numpy array
def saveNumpy(array, fileName):
    numpy.save(fileName, array)
        
# Takes the filename and returns the array that is contains after loading file
def loadNumpy(fileName):
    data = numpy.load(fileName)
    return data 


# Helper Function - Linear Filters
# Takes a square kernel of any size and an image as parameters, returns the image resulting from the kernel being applied
def applyLinearFilter(squareKernel, image):
    ddepth = -1 #same depth as original image
    filteredImage = cv2.filter2D(image, ddepth, squareKernel)
    return filteredImage

# Helper Function - Mean Filtering
# Takes an image and the size of the image filter, returns the image after the filter has been applied
def applyMeanFilter(image, filterSize):
    filterSize = (filterSize, filterSize) 
    filteredImage = cv2.blur(image, filterSize)
    return filteredImage

# Helper Function - Generate Ellipse on slice
# Generate a random ellipse, takes the height width and depth of the desired ellipse shape
def generateEllipse(axisH, axisW, xCenter, yCenter, depth = 8):

    color = (2**depth) - 1
    simulatedImage  = numpy.zeros((224, 224), numpy.uint8)
    angles = numpy.linspace(0, 360, 360 * 2)

    for i in angles:
        currentAngle = i * numpy.pi/180
        yPos = xCenter + axisW * numpy.sin(currentAngle)
        xPos = yCenter + axisH * numpy.cos(currentAngle)
        
        xPos = numpy.clip(xPos, 0, 223)
        yPos = numpy.clip(yPos, 0, 223)

        row, column = int(xPos), int(yPos)
        
        if i > 270:
            simulatedImage[yCenter:row, column:xCenter] = color
        elif i > 180:
            simulatedImage[row:yCenter, column:xCenter] = color
        elif i > 90:
            simulatedImage[row:yCenter, xCenter:column] = color
        elif i > 0:
            simulatedImage[yCenter:row, xCenter:column] = color

    simulatedImage = simulatedImage.astype("uint8")
    return simulatedImage

# Helper Function - Provided by professor from assignment 1
# Takes the volume and the max depth and maps the existing pixels to the range specified by the new depth
def resampleDepth(volume,maxDepth = 255):
    originalDepth = numpy.max(volume)
    newImage = volume.copy()
    newImage = numpy.divide(newImage,originalDepth)
    newImage *= maxDepth
    newImage = newImage.astype("uint8")
    return newImage

def testQ1to5():
    # Question 1
    ellipseVolume = generate3DEllipse(224, 224, 224, 75, 80)
    saveNumpy(ellipseVolume, 'simulatedTumor.npy')
    
    # Used to test data on the same dataset multiple times and produce results in pdf
    # ellipseVolume = loadNumpy('FINALsimulatedTumor.npy')
    
    # Question 2
    reslice(ellipseVolume, 112, 112, 112)

    # # Question 3
    intensity = maxIntensity(ellipseVolume)
    displayResampledSlices(intensity, "Max Intensity")
    radiograph = digitalRadiograph(ellipseVolume)
    displayResampledSlices(radiograph, "Radiograph")

    # # Question 4
    pointBased(ellipseVolume, 60, 100)

    # Question 5
    y = ellipseVolume.shape[0] // 2
    imageCoronal = ellipseVolume[:,y,:]
    imageCoronal = resampleDepth(imageCoronal)
    interest = numpy.unravel_index(numpy.argmax(imageCoronal), imageCoronal.shape)
    background = numpy.unravel_index(numpy.argmin(imageCoronal), imageCoronal.shape)
    newImage = regionGrowing(imageCoronal, [interest, background], 0, {interest: 1, background: 0})
    displayMatPlot(newImage, "Region Growing Threshold")


def testQ6():
    # Part A: Slices used were Axial 86, Coronal 105 and Sagittal 139
    brain = loadNumpy('MRBrainTumour.npy')
    reslice(brain, 86, 105, 139)

    # # Part B: Image Rendering
    intensity = maxIntensity(brain)
    displayResampledSlices(intensity, "Max Intensity")
    radiograph = digitalRadiograph(brain)
    displayResampledSlices(radiograph, "Radiograph")

    imageAxial = brain[86,:,:]
    imageAxial = resampleDepth(imageAxial)

    # Attempt to improve segmentation
    smoothingFilter = (1/10.0) * numpy.array([[1,1,1],[1,2,1],[1,1,1]])
    # edgeFilter = numpy.array([[0,0,0],[0,-1,1],[0,0,0]])
    # sharpeningFilter = numpy.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # imageAxial = applyLinearFilter(sharpeningFilter, imageAxial)
    # imageAxial = applyLinearFilter(smoothingFilter, imageAxial)
    # imageAxial = applyMeanFilter(imageAxial, 7)

    pointBased(imageAxial, 75, 105)

    # Testing the difference seeds
    # interest = numpy.unravel_index(numpy.argmax(imageAxial), imageAxial.shape)
    # background = numpy.unravel_index(numpy.argmin(imageAxial), imageAxial.shape)
    # newImage = regionGrowing(imageAxial, [interest, background], 0, {interest: 1, background: 0})
    # displayMatPlot(newImage, "Region Growing Threshold with max and min points")
    # newImage = regionGrowing(imageAxial, [(107,142), (150, 98)], 0, {(107,142): 1, (150, 98): 0})
    # displayMatPlot(newImage, "Region Growing Threshold with specific tumour points and threshold 0")
    newImage = regionGrowing(imageAxial, [(107,142), (150, 98)], 6, {(107,142): 1, (150, 98): 0})
    displayMatPlot(newImage, "Region Growing Threshold")

def main():
    # In order to see the analysis performed on the 3d ellipse testQ1to5 is used
    testQ1to5()

    # In order to see the analysis performed on the brain tumour testQ6 is used
    testQ6()


if __name__ == "__main__":
    main()
