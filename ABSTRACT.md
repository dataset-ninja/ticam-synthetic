**TiCaM Synthectic Images: A Time-of-Flight In-Car Cabin Monitoring Dataset** is a time-of-flight dataset of car in-cabin images providing means to test extensive car cabin monitoring systems based on deep learning methods. The authors provide a synthetic image dataset of car cabin images similar to the real dataset leveraging advanced simulation software’s capability to generate abundant data with little effort. This can be used to test domain adaptation between synthetic and real data for select classes. For both datasets the authors provide ground truth annotations for 2D and 3D object detection, as well as for instance segmentation.

Note, similar **TiCaM Synthectic Images: A Time-of-Flight In-Car Cabin Monitoring Dataset** dataset is also available on the [DatasetNinja.com](https://datasetninja.com/):

- [TiCaM Real Images: A Time-of-Flight In-Car Cabin Monitoring Dataset](https://datasetninja.com/ticam-real-images)

## Dataset description

With the advent of autonomous and driver-less vehicles, it is imperative to monitor the entire in-car cabin scene in order to realize active and passive safety functions, as well as comfort functions and advanced human-vehicle interfaces. Such car cabin monitoring systems typically involve a camera fitted in the overhead module of a car and a suite of algorithms to monitor the environment within a vehicle. To aid these monitoring systems, several in-car datasets exist to train deep leaning methods for solving problems like driver distraction monitoring, occupant detection or activity recognition. The authors present TICaM, an in-car cabin dataset of 3.3k synthetic images with ground truth annotations for 2D and 3D object detection, and semantic and instance segmentation. Their intention is to provide a comprehensive in-car cabin depth image dataset that addresses the deficiencies of currently available such datasets in terms of the ambit of labeled classes, recorded scenarios and provided annotations; all at the same time.

<img src="https://github.com/dataset-ninja/ticam-synthetic/assets/120389559/2050b51f-04d9-4aae-b4f4-798680f68e8d" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Synthetic images. RGB scene, depth image with 2D bounding box annotation, segmentation mask.</span>

A noticeable constraint in existing car cabin datasets is the omission of certain frequently encountered driving scenarios. Notably absent are situations involving passengers, commonplace objects, and the presence of children and infants in forward and rearward facing child seats. The authors have conscientiously addressed this limitation by capturing a diverse array of everyday driving scenarios. This meticulous approach ensures that our dataset proves valuable for pivotal automotive safety applications, such as optimizing airbag adjustments. Using the same airbag configuration for both children and adults can pose a fatal risk to the child. Therefore, it is imperative to identify the occupant class (*person*, *child*, *infant*, *object*, or *empty*) for each car seat and determine the child seat configuration (forward-facing FF or rearward-facing RF). The dataset also includes annotations for both driver and passenger activities.  

It is easier to generate realistic synthetic depth data for training machine learning systems than it is to generate RGB data. Compared to real datasets, creation and annotation of synthetic datasets is less time and effort expensive. Recognizing this fact has led authors to also create a synthetic front incar cabin dataset of over 3.3K depth, RGB and infrared images with same annotations for detection and segmentation tasks. The authors believe this addition makes their dataset uniquely useful for training and testing car cabin monitoring systems leveraging domain adaptation methods.

## Synthetic Data Generation

The authors render synthetic car cabin images using 3D computer graphic software Blender 2.81. The 3D models of Mercedes A Class is from [Hum3D](http://www.hum3d.com), the everyday objects were downloaded from [Sketchfab](http://www.sketchfab.com) and the human models were generated via [MakeHuman](http://www.makehumancommunity.org). In addition, High Dynamic Range Images (HDRI) was used to get different environmental backgrounds and illumination, and finally, in order to define the reflection properties and colors for the 3D objects, textures from [Textures](http://www.textures.com) were obtained for each object.

In pursuit of simulating driving scenarios, the authors consistently designated the occupant of the driver's seat as an adult in a driving pose. For the passenger seat, the occupant was randomly chosen from the remaining classes, including child seats (both empty and child-occupied), everyday objects, and adult passengers. While some objects, such as handbags, backpacks, and bottles, were selected to mirror those found in real driving scenarios, they were grouped under the overarching class *everyday object*. Similarly, children and infants were collectively categorized under the class *person*.

The authors recreated the actions participants were instructed to perform during the recording of the actual dataset. Given that driver poses are naturally constrained by the car elements they interact with, and to prevent any overlap, the authors established fixed poses for hand positions that align with possible real driving scenarios. Meanwhile, they allowed for movement in other body parts up to a specified threshold.

<img src="https://github.com/dataset-ninja/ticam-synthetic/assets/120389559/5706bc09-6991-4bac-9370-6247be01d700" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;"> Different scenarios in the synthetic dataset.</span>

##  Data Format

* **Depth Z-image.** The depth image is undistorted with a pixel resolution of 512 × 512 pixels and captures a 105◦ × 105◦ FOV. The depth values are normalized to [1mm] resolution and clipped to a range of [0, 2550mm]. Invalid depth values are coded as ‘0’. Images are stored in 16bit PNG-format.

* **IR Amplitude Image.** Undistorted IR images from the depth sensor are provided in the same format as the depth image above.

* **RGB Image.** Undistorted color images are saved in PNG-format in 24bit depth. The synthetic RGB images have the same resolution and field of view as the corresponding depth images (512 × 512).

* **2D bounding boxes.** Each 2D box in the synthetic dataset is represented by its class ID, the top-left and the bottom-right corners.

* **Pixel segmentation masks.** For synthetic images the authors provide a single mask.
