# Dataset Summary for Report

## 1. Dataset used in this project
- Dataset root: `data/`
- Total images: **24,892**
- Total bounding boxes (labels): **53,564**
- Train/Val/Test ratio: **80/10/10**
- Number of defect classes: **6**

## 2. Defect classes
1. missing_hole
2. mouse_bite
3. open_circuit
4. short
5. spur
6. spurious_copper

## 3. Split statistics
| Split | Images | Label files | Bounding boxes |
|---|---:|---:|---:|
| train | 19,913 | 19,913 | 42,828 |
| val | 2,489 | 2,489 | 5,341 |
| test | 2,490 | 2,490 | 5,395 |

## 4. Data formats used
- YOLO TXT labels: `data/<split>/labels/`
- Pascal VOC XML annotations: `data/<split>/annotations_xml/`
- COCO JSON annotations: `data/annotations_json/{train,val,test}.json`

## 5. Image count by class
| Class | Images |
|---|---:|
| missing_hole | 3,945 |
| mouse_bite | 4,029 |
| open_circuit | 3,823 |
| short | 3,822 |
| spur | 3,854 |
| spurious_copper | 3,863 |

## 6. Bounding box summary
- Overall bbox count: **53,564**
- Width range: **0.004687 - 0.489781**
- Height range: **0.003125 - 0.403627**
- Mean bbox width: **0.044806**
- Mean bbox height: **0.045614**
- Mean bbox area: **0.002166**

## 7. Source files used for this summary
- `dataset_info.json`
- `summary.json`
- `classes.txt`

Generated from project metadata snapshot dated: **2026-03-26 (UTC)**.
