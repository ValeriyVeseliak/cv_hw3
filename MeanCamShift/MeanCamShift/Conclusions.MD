During this work, i used OpenCV to implement tracker based on MeanShift and CamShift algorithms.
These algorithms were tested on several test datasets, like Biker, BlurBody, Dog, DragonBaby, Jump and Surfer.

1 ) MeanShift has shown pretty good results on images and frames, when the object we are trying to track is different from background.
In this case we can see good tracking accuracy. But as soon as we have similar colors on background or object on the movement path, which have similar colors with the tracking object,
we can see that our algorithm just stop on this object or start fluctuating around and loosing accuracy. For example, as in dataset Jump or Dog, where we can see, that the sportsman is pretty similar with some pillar,
so when the tracking rectangle is around it, it looses the tracking object.
So MeanShift has shown good results on such test datasets as Surfer, DragonBaby, BlurBody (wors very well, even with huge camera instability), Biker.

To run MeanShift script use command:
```bash
python3 MeanShift.py --images_folder ../Surfer/img/ --roi_path ../surfer_roi.png
```

2 ) CamShift algorithm was used to check tracking abilities and understand how we can fit the tracking window to the changing object.
CamShift was also tested on the same datasets and shown worse accuracy, than MeanShift because of exploding window issue.
So during the implementation and testing i've faced with issue, when the tracking window explodes and become of the size of whole image.
It can be cause of similarity of tracking object and background, so during the comparison we can see, that background is also recognized as tracking object.
This issue was seen on such datasets as Biker, Jump, DragonBaby, Dog(tracking object changed to human).
It shown good accuracy only on Surfer dataset.
Also i've tried to fix this issue by fitting mask of tracking object and creating more specific histogram, that's why you can find several configs for mask in CamShift script.

To run CamShift script use command:
```bash
python3 CamShift.py --images_folder ../Surfer/img/ --roi_path ../surfer_roi.png
```

So according to the comparison i see MeanShift algorithm more stable and it has no problems with tracking window, as it doesn't change its size.

P.S. Haven't added datasets to zip as it would make it huge.