# image-classification-with-svm

Tested with images scrapped from flickr. Tags used were Sensational and Drab. 

Features extracted (Total dimensions: 1631) were:

### Color	
- BGR color histogram (3 * 64 = 192 dimensions)
- HSV color histogram (16 * 4 * 4 = 256 dimensions)
- L*a*b color histogram ( 4 * 14 * 14 = 784 dimensions)

### Shape	
- Line: number of straight lines (1 dimension)
- Line: percentage of parallel lines (1 dimension)
- Line: histogram of line orientations (9 dimensions)
- Line: histogram of line distances (6 dimensions)
- Corner: percentage of global corners (1 dimension)
- Corner: percentage of local corners	(1 dimension)
- Edge Orientation Histogram (64 dimensions)
- Histogram of Oriented Gradients (225 dimensions)

### Texture	
- Local Binary Pattern (59 dimensions)
- Gabor (32 dimensions)

# Accuracy

| Color | Shape | Texture | Combined |
| --- | --- | --- | --- |
| 0.786 | 0.696 | 0.695 | 0.822 |


