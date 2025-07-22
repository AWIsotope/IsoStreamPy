### IsoStreamPy
## Data evaluation of heavy stable Isotopes

The ISOtope STREAMlit PYthon software can be used for data evaluation of heavy stable isotopes measured with a Thermo Scientific Neptune. This software was originally written for the data evaluation and quality check of Cu, Sn and Sb isotopes, mostly used in the archeometallurgy. It was created to allow a fast data evaluation with C-SSBIN and Baxter. 

# !!! ATTENTION !!! The software is in an early alpha-stage and only works with special cup configurations of the neptune. 

Feel free to contact for further assistance or if you would like to join and help me to make this software usable to more setups and isotopes.

Installation:

download folder, install the python packages listed in the requirements.txt and run with "streamlit run IsoStreamPy.py". 

Usage: 
Before starting of the sequence of the neptune, the checkbox "online evaluation" must be ticked. Afterwards, the data must be "exported". 
For Cu: The bracketing Standard must be names as "Sxx", while xx ist a number between 0 and 99. However, it must start with "S01".
For Sb and Sn: The bracketing Standard must be names as "Nisxx", while xx ist a number between 0 and 99. However, it must start with "Nis01".
Blanks: Name them as "Blk".

Then upload the folder to the "input" folder. 
The first and last file *MUST* be a Blank.

Hint: Restart IsoStreamPy with the button on the interface after each change of Isotope system and/or folder. 


As already mentioned before, this is an early alpha stage and was created for the usage in the archaeometallurgy. 







