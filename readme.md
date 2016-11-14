POS Tagging for CMISM, ICON 2016
===================


Please find the details of the shared task [here](http://ltrc.iiit.ac.in/icon2016/).

----------


Running the code
-------------
- Install [CRF++](https://taku910.github.io/crfpp/) and [pycrfsuite](https://python-crfsuite.readthedocs.io/) before execution.
- Go to Resources/training_data and testing_data to add your respective files.
- Sample Train File name : domainName_langPair_FinerOrCoarser.txt, FB_HI_EN_FN.txt
- Sample Test File name : domainName_langPair_Test_Raw.txt, FB_HI_EN_Test_Raw.txt
- Make sure to add the read files with different names in the *main_train_frame* in **main.py**
- Enter the language pair, mode and Classifier mode in main.py. Classifier modes are **crf++** and **pycrf**.
- For final testing, we only used the **crf++** module as their CV results were better than pycrfsuite.
- Run **main.py**.

Reporting Doubts and Errors
-------------------
- For any queries, please contact me at **stripathi1770@gmail.com**.

- SMPOST toolbox to be announced soon.

- Please refer to the publication for detailed results.
