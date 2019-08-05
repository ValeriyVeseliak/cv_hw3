1 ) First what i did, implemented template matching with OpenCV to understand how it works and what parameter takes.
As in OpenCV there is no SAD based template matching, i did only 2 metrics: SSD and NCC.
Both metrics works okay for single image template matching. 

To run template matching with OpenCV use command:
```bash
python3 template-matching-opencv.py --image ../Surfer/img/0001.jpg --template ../surfer_roi.png --method SSD
```

2 ) Next I've implemented template matching for sequence of images. It works normally for SSD for almost all part of time, 
but for NCC it works much worse.
Looks like template matching is not so good for sequences of frames which are changing during time (only if the tracked object isn't changing much).

To run sequential template matching use command:
```bash
python3 template-matching-opencv-sequential.py --images_folder ../Surfer/img/ --template ../surfer_roi.png --method NCC
```


3 ) I've tried to implement template matching manually without usage of OpenCV library.
First issue i've faced with is performance, as my implementation works suuuuper slowly.
In comparison with OpenCV implementation it works in tens times worse.
Also i've tried to do sequential, but it works very slowly and calculation of tracking window on new frame takes several seconds.

To run manual you can use command:
```bash
# for single image
python3 template-matching-manual.py --image ../Surfer/img/0001.jpg --template ../surfer_roi.png --method SAD

# for sequence of images
python3 template-matching-manual-sequential.py --images_folder ../Surfer/img/ --template ../surfer_roi.png --method SAD
```


4) Lucas-Kanade was implemented with help of OpenCV (as specified in task) in two ways:
- Tracking specified window(similar to previous problem). In this case we track two points of our rectangle and it works pretty bad, as we can simply
lose only one point and tracking is broken (the same as when you try to test it with Biker dataset). But it also works not bad on Surfer dataset, where
we have strong difference between patch we are looking for and background.
To run this tracking you command:
```bash
python3 Lucas-Kanade-Tracker.py --images_folder ../Surfer/img/ --roi_path ../surfer_roi.png
```
- Tracking of features within specified patch. Here we use Shi-Tomasi method (OpenCV implementation) to find corners as features to track and then check is it tracking good or not.
Here we use template matching to find a tracking window, than create a mask and take features only from part of the images, that is inside the mask rectangle.
This approach works pretty well except of some cases, where we lose our tracking objects when background become similar to tracking points or tracking points are changed a lot.
Also we can see, that it works worse, when we have sharp and quick movements as in DragonBaby dataset. In this case we need to use pyramid-extension of LC algorithm.
To run this kind of Lucas-Kanade tracker use command:
```bash
python3 Lucas-Kanade-MaskFeatures-Tracker.py --images_folder ../Dog/img/ --roi_path ../dog_roi.png
``` 
- Pyramidal extension of Lucas-Kanade was implemented with use of the same method, but including new parameter of 'pyramids_number'.
This approach gives ability to tack features even when tracking points was moved far from previous location.
So we use 'maxLevel' parameter to specify number pyramids and if we set it to 0, pyramids are not used (single level), if set to 1, 
two levels are used, and so on; if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel.
So with usage of pyramids we can see additional accuracy on tracking.
To use pyramidal extension of Lucal-Kanade use the command:
```bash
python3 Lucas-Kanade-Pyramidal-MaskFeatures-Tracker.py --images_folder ../DragonBaby/img/ --roi_path ../baby_roi.png --pyramids_number 3
```

I had another idea how to implement tracking with Lucas-Kanade algorith, but because of the lack of time and confidence that this approach will work i will just add its description.
So the idea is to create patch, retrieve features from this patch and calculate 4 edge features, which are the nearest to the edges of the patch.
Next step is to calculate the distance to all the edges. Then when we track the optical flow with LC we can just recreate a rectangle with adding the same distance to 4 edge features.
This approach is only as an idea for future implementation.


All the implementations were tested on the same datasets as MeanShift and CamShift.
Using Lucas-Kanade we can see, that it has better performance, that MeanShift and CamShift, but also can have error in tracking because of big error between correspondent features, 
quick movement of objects and features, features disappearing from frames.