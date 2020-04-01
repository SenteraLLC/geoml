# research_tools

API to process and analyze all of the historical data collected from the Rosen lab.

The overall goal is to determine the baseline prediction accuracy we should expect for each response variable (e.g., petiole nitrate, vine N, biomass, etc.) for a given set of input features (e.g., cropscan bands, hyperspectral imagery, weather, etc.).

## Classes
There [will be] multiple classes that work together to achieve the desired behavior/results. Here is a brief summary:

### join_tables
Assists with joining tables that contain training data. In addition to the join, many of the user functions available add new columns/features to the input DataFrame that hopefully explain the response variable being predicted.
