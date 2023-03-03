# Welcome to Streamlit!

# Image Similarity Detection

This is a project that uses computer vision to determine the similarity between images. Specifically, we use machine learning algorithms to automatically find similarities and differences between two images, and then display these results on a web page using the Streamlit library.

### Installation Dependencies
To run this project, you need to install the following dependencies:
```sh
Python 3.6+
streamlit==1.4.0
numpy
pillow
scikit-image
altair
scipy
tensorflow_hub
opencv-contrib-python-headless
tensorflow==2.7.0
pandas
```

You can install them by running the following command:
```sh
pip install -r requirements.txt
```

### How to Use
You can run the project by executing the command:
```sh
streamlit run app.py
```

After running, you can upload two images by clicking on the corresponding button, and then click on the "Compare" button to see the results. You will see the similarity measure between the two images based on the Pearson correlation coefficient, which will be displayed on the page as a number from 0 to 1.

### Example Images

Example images for testing are located in the test_img folder, including the files a (3).png and a (1).png.

### Authors

This project was created by Alexey Kh.