Dataset with 12,086 real estate properties for sale in southern Spain in April 2024. It includes the following files:
- properties.csv - comma separated values for 9 data fields
- descriptions.tar.gz - compressed archive with a file for each property including a textual description of it
- images.tar.gz - compressed archive including a folder with images for each property

The fields in the CSV are ordered as follows:
reference, location, price, title, bedrooms, bathrooms, indoor surface area in sqm, outdoor surface area in sqm, pipe-separed list of features of the property

The descriptions files are extracted into a "descriptions" folder and inside there is a file with the reference of each property in the CSV as the filename and .txt as extension.

The image files are extracted into a "images" folder and inside  there is a folder for each property with the folder name being the reference of the property in the CSV file. The images have been resized to 300px as maximum dimension, keeping the aspect ratio.

