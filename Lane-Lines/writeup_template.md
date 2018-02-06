# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image0]: ./img/0.png "Initial"
[image1]: ./img/1.png "Grayscale"
[image2]: ./img/2.png "Blurred"
[image3]: ./img/3.png "Canny"
[image4]: ./img/4.png "Interest"
[image5]: ./img/5.png "Lines"
[image6]: ./img/6.png "Color merge"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

This is the pipeline I used on each image on the video:

Example initial image:
![alt text][image0]


1. I converted the image to grayscale to reduce the size (I'm eliminating depth) and to not consider colors. I used the function `grayscale`.

![alt text][image1]

2. I've applied a Gaussian blur to diffuse the image. The Gaussian blur uses a kernel size. I've tried some. Finally, I've used `kernel_size=5` with the function `gaussian_blur`.

![alt text][image2]

3. After that, I've detected the edges using a Canny Edges algorithm specifying a lower and higher threshold. I used the `canny` function with `low_threshold=100` and `high_threshold=255`.

![alt text][image3]

4. One I had the edges, I've selected a region of interest. I took the car vision, and tried to draw an area that contains the lane lines. After a lot of trying, I've selected the vertices: (110, xsize), (ysize/2-20,310) , (ysize/2+20,310),(900,xsize). I used the method `region_of_interest` over my masked edges image.

![alt text][image4]

5. Finally, I will dectect and draw the lines using the `hough_lines` method. I used `rho=1`,`theta=np.pi/180`, `min_line_len=200`, `threshold=15`, and `max_line_gap=300`.

![alt text][image5]

6. Created a color image and merge it with the original image using `weighted_img` function

![alt text][image6]


### 2. Identify potential shortcomings with your current pipeline

If the lane is wider, the lines could fall outside the region of interest and not being detected.

If there is another drawing on the lane, it could be confused by the lane delimiters.

My pipeline doesn't draw a perfect line. If the inner smaller lines are too separated on the same lane line, it could be confused as not line at all.

It paints the car in front in some frames of the video.

### 3. Suggest possible improvements to your pipeline

Improve the `draw_line` method detecting slope and intersection of the pair of points to draw a continous line.

Don't use region of interest, just took the car vision and detect to the left and to the right, the next pixel that can belong to a line.

Better parameter tunning to avoid the car in front.