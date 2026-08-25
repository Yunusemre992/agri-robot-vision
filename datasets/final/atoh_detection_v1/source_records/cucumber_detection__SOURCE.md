Cucumber detection staging source

Annotated source: https://universe.roboflow.com/thomas-j6iuv/cucumber_disease-6/dataset/2
Original image dataset: https://data.mendeley.com/datasets/y6d3z6f8z9/1
License: CC BY 4.0 (Attribution 4.0 International)

Accepted source-class mapping:
0 Anthracnose -> 12 cucumber_leaf_diseased
2 Belly-Rot -> 3 cucumber
3 Downy-Mildew -> 12 cucumber_leaf_diseased
4 Fresh-Cucumber -> 3 cucumber
5 Fresh-Leaf -> 11 cucumber_leaf_healthy
6 Gummy-Stem-Blight -> 12 cucumber_leaf_diseased
7 Pythium-Fruit-Rot -> 3 cucumber

Excluded:
1 Bacterial-Wilt: annotations surround whole wilted plants/clusters rather than
individual leaves, so they are incompatible with cucumber_leaf_diseased.

The Roboflow archive is an annotated/augmented derivative of the Mendeley
dataset. Do not merge the unannotated original ZIP into training, because that
would duplicate source images.
