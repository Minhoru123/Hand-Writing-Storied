Data Exploration and Preparation

For this I thought first of:

- Having three main parts

  - The data organization where I had a dataclass to store all the img stats such width, height, the filename
  - The image analysis function helps with the dataset and extracts all key metrics like dimensions and how much coverage. It also has some error handling for images that can't be processed
  - Then with all this information we crate a detailed report for the 100 png snippets and another general smaller one with general stats.
- The analysis_report.txt  has

  * Total number of images
  * Average dimensions
  * Grayscale vs. color distribution
  * Text coverage statistics
- And the detailed_stats.csv has all of the same abover but just per image.
