import cv2
import numpy as np

# Create a black image of size 500x500
img = np.zeros((500, 500, 3), np.uint8)

# Define an array of 50 points
points = np.array([[100, 100], [200, 200], [300, 300], [400, 400], [100, 400], [200, 300], [300, 200], [400, 100], [250, 250], [150, 150], [350, 350], [450, 450], [150, 450], [250, 350], [350, 250], [450, 150], [250, 150], [350, 450], [150, 250], [450, 350], [100, 200], [200, 400], [400, 200], [200, 100], [300, 150], [150, 300], [300, 450], [450, 300], [350, 150], [150, 350], [350, 400], [400, 350], [250, 200], [200, 250], [250, 400], [400, 250], [150, 200], [200, 150], [400, 450], [450, 400], [150, 400], [400, 150], [300, 200], [200, 300]], np.int32)
point1 = np.array([[367, 279],
 [368, 279],
 [368, 279],
 [368, 278],
 [369, 278],
 [369, 278],
 [370, 278],
 [371, 278],
 [372, 278],
 [372, 278],
 [370, 279],
 [371, 280],
 [371, 279],
 [371, 280],
 [370, 281],
 [370, 281],
 [371, 280],
 [371, 281],
 [371, 281],
 [371, 281],
 [371, 281],
 [371, 281],
 [370, 282],
 [370, 282],
 [369, 283],
 [369, 283],
 [369, 283],
 [369, 283],
 [369, 284],
 [369, 284],
 [369, 285],
 [369, 286],
 [369, 286],
 [369, 286],
 [369, 286],
 [369, 286],
 [369, 286],
 [369, 287],
 [368, 287],
 [368, 288],
 [368, 288],
 [367, 289],
 [368, 289],
 [368, 289],
 [368, 290],
 [367, 291],
 [367, 292],
 [367, 292],
 [367, 292],
 [367, 293],
 [366, 295],
 [367, 297],
 [368, 297],
 [368, 298],
 [368, 298],
 [367, 299],
 [366, 300],
 [366, 301],
 [366, 302],
 [366, 303],
 [366, 305],
 [366, 305],
 [366, 306],
 [365, 307],
 [365, 307],
 [366, 308],
 [366, 310],
 [366, 310],
 [366, 312],
 [366, 312],
 [366, 314],
 [366, 315],
 [366, 316],
 [367, 317],
 [366, 319],
 [366, 321],
 [365, 322],
 [364, 324],
 [364, 326],
 [364, 326],
 [364, 328],
 [364, 330],
 [364, 332],
 [364, 334],
 [364, 335],
 [364, 337],
 [364, 339],
 [364, 342],
 [363, 343],
 [363, 344],
 [363, 346],
 [363, 348],
 [363, 351],
 [363, 353],
 [363, 355],
 [363, 357],
 [363, 359],
 [362, 361],
 [362, 363],
 [362, 365],
 [362, 367],
 [361, 368],
 [361, 370],
 [361, 371],
 [361, 373],
 [361, 375],
 [361, 375],
 [361, 376],
 [360, 377],
 [360, 379],
 [359, 381],
 [358, 382],
 [358, 383],
 [357, 384],
 [356, 384],
 [355, 386],
 [355, 387],
 [355, 387],
 [354, 387],
 [353, 388],
 [352, 389],
 [351, 390],
 [351, 390],
 [350,392],
                   [350, 392],
 [350, 392],
 [351, 393],
 [351, 394],
 [351, 394],
 [351, 394],
 [351, 393],
 [351, 394],
 [352, 393],
 [352, 393],
 [352, 393],
 [352, 393],
 [352, 392],
 [352, 392],
 [352, 392],
 [351, 391],
 [352, 390],
 [352, 390],
 [352, 389],
 [352, 389],
 [353, 388],
 [353, 387],
 [354, 387],
 [354, 387],
 [354, 386],
 [354, 385],
 [354, 385],
 [354, 384],
 [353, 383],
 [353, 382],
 [353, 381],
 [354, 381],
 [355, 380],
 [355, 379],
 [354, 379],
 [354, 378],
 [354, 378],
 [354, 377],
 [354, 377],
 [353, 377],
 [353, 377],
 [353, 376],
 [353, 376],
 [353, 375],
 [353, 375],
 [352, 374],
 [352, 374],
 [352, 373],
 [352, 372],
 [352, 372],
 [352, 370]])


# points = np.concatenate(point1)
# Reshape the points array to be Nx1x2, where N is the number of points
points = point1.reshape(-1, 1, 2)
# print(point)

# Draw the path in blue color with thickness of 5
img = cv2.polylines(img, [points], False, (255, 0, 0), 5)

# Display the image
cv2.imshow("Path", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
