# Problem Description
Sudoku is one of the most popular puzzles of all time. The goal of Sudoku is to fill a 9x9 grid
such that each row, each column and 3x3 grid contains all of the digits between 1 to 9.In this
project we aim to create a real time Sudoku solver which recognizes the elements of Sudoku
puzzles and provides a digital solution using Computer vision.Sudoku Solver is the collection
of very basic image processing techniques. A very good way to start is the OpenCV library
which can be compiled on almost all the platforms.

<h2>Software Requirements</h2>
<li>OpenCV-for handling image processing</li>
<li>Numpy-for handling numeric data</li>
<li>Tersseract- for optical character recognition (OCR) tool</li>

<h2>Algorithmic Approach</h2>
Majorly there are <b>Three main steps</b>to solve our problem.
<br>
<ol>
  <li><b>Extracting Sudoku grid from our webcam Image</b></li>
    The first one is to extract the Sudoku grid from our webcam Image and Fig 1 shows how to proceed step by step in achieving this task.
    
  <p align="center">
   <img src="https://github.com/aaru9dua/Computer-vision/blob/main/Real-time%20Sudoku%20Solver/step1.png" width="900" height="450" /></p>
  <br>
  <li><b>Detecting Digits from Extracted Image</b></li>
  The second step is to preprocess the extracted image in order to Detect the Digits using pytesseract. Fig 2 shows the steps involved to detect the grid numbers.
  
  <p align="center">
   <img src="https://github.com/aaru9dua/Computer-vision/blob/main/Real-time%20Sudoku%20Solver/step2.png" width="900" height="450" /></p>
  <br>
  <li><b>Solve the Sudoku puzzle using recurssive method</b></li>
</ol>

<h2>RESULT</h2>

![Media1](https://user-images.githubusercontent.com/46483403/122409778-0a095480-cfa1-11eb-8f17-77583f6495e1.gif)
![Picture1](https://user-images.githubusercontent.com/46483403/122447789-b9eeba00-cfc1-11eb-9751-424cd8cfd2bd.png)
