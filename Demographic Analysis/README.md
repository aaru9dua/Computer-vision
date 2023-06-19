# Computer Vision Semester Project - Detecting Demographics of People in Images Using Computer Vision  
##### by Aarushi Dua, Ganesh Arkanath, Rasika Muralidharan, Sai Teja Burla

## Dependencies
<ul>
  <li>numpy</li>
  <li>pandas</li>
  <li>scikit-learn</li>
  <li>scikit-image</li>
  <li>matplotlib</li>
  <li>tensorflow</li>
  <li>opencv</li>
  <li>dlib</li>
</ul>

## Model workflow

![image](https://github.com/aaru9dua/Computer-vision/assets/46483403/713a10af-6c6e-4e1a-a0c4-4a3d19979745)

## How to run
<ol>
  <li> Download the dataset zip file from <a href="https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view">Here</a>.</li>
  <li> Unzip the folder and move both <i>train</i> and <i>val</i> folders to the root of the project directory.</li>
  <li> The CSV files containing the labels for the dataset are included in the project directory and need not be downloaded separately. They were downloaded from <a href="https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view">Train</a> and <a href="https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view">Validation</a>.</li>
  
  <li>
  
  ``` python custom_model.py ```
  
  ``` python custom_model_no_lbp.py ```
  
  These commands will run the respective models which will give a comparative analysis of the different methods employed.
  </li>
  
  <li> Once completed, the model will be saved in the project's root directory along with a .txt file containing the model's learning history. Two graphs will also be generated, one for loss and one for accuracy, of the model across all 3 measures - race, gender, and age.</li>
</ol>

#### Optional
The geometric facial features have already been extracted and stored in csv files, which are used in both the models. The code for extracting the facial features from images is contained in the file _geodesic.py_. To run the file, execute the following command -

``` python geodesic.py ```

This will generate two csv files, one for training data and one for validation data. These two files are required to run the models. Since these two files have already been generated, this step is not required to run the models.


## Credits
https://github.com/dchen236/FairFace for the dataset
