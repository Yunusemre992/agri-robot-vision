# ATOH Agriculture Detection Dataset v1 - Sources

This dataset is a derived academic/internship prototype. Each upstream source
retains its own license. The combined dataset must not be redistributed until
all source licenses, especially deepNIR and PlantDoc, are independently verified.

## Active class schema

- `0`: `apple`
- `1`: `tomato`
- `2`: `bell_pepper`
- `3`: `cucumber`
- `4`: `apple_leaf_healthy`
- `5`: `apple_leaf_diseased`
- `6`: `tomato_leaf_healthy`
- `7`: `tomato_leaf_diseased`
- `8`: `bell_pepper_leaf_healthy`
- `9`: `bell_pepper_leaf_diseased`
- `10`: `cucumber_leaf_healthy`
- `11`: `cucumber_leaf_diseased`

## Reserved for dataset v2 (not trained in v1)

- staging `4`: `goji_berry`
- staging `13`: `goji_berry_leaf_healthy`
- staging `14`: `goji_berry_leaf_diseased`

## Upstream sources

### MinneApple

- Staging key: `minneapple`
- Official/reference URL: https://rsn.cs.umn.edu/index.php/MinneApple
- License: CC BY-NC-SA 3.0 US
- Notes: Orchard apple detection source; original masks converted to bounding boxes.

### Laboro Tomato

- Staging key: `laboro_tomato`
- Official/reference URL: https://github.com/laboroai/LaboroTomato
- License: CC BY-NC-SA 4.0
- Notes: Greenhouse tomato instance annotations converted to bounding boxes.

### deepNIR apple and capsicum subset

- Staging key: `kaggle_gpt`
- Official/reference URL: https://zenodo.org/records/6324489
- License: NOT VERIFIED on the official Zenodo record
- Notes: The staged subset was proven byte-identical, image and label, to the official deepNIR apple/capsicum subset (185/185). The staging folder keeps its historical name kaggle_gpt for reproducibility.

### PlantDoc detection subset

- Staging key: `plantdoc_detection`
- Official/reference URL: https://github.com/pratikkayal/PlantDoc-Dataset
- License: See copied PlantDoc SOURCE_LICENSE.txt; verify before redistribution
- Notes: Apple, tomato and bell-pepper healthy/diseased leaf boxes.

### Cucumber detection merged subset

- Staging key: `cucumber_detection`
- Official/reference URL: See copied staging SOURCE.md
- License: CC BY 4.0 according to staging SOURCE.md
- Notes: Cucumber fruit plus healthy/diseased cucumber leaf boxes.

### Tomato Leaf Image Dataset for Disease Analysis in Real-World Environment

- Staging key: `tomato_realworld`
- Official/reference URL: https://data.mendeley.com/datasets/rnbsw72zb5/1
- License: NOT VERIFIED on the official Mendeley record
- Notes: Bangladesh field-collected tomato leaves with complex backgrounds. Ships with ready YOLO boxes (H=0, D=1) remapped to staging 7/8. Source train/val split preserved as train/valid.
