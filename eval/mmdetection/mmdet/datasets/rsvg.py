from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class RSVGDataset(CocoDataset):
    
    CLASSES = ('vehicle','chimney','golffield','Expressway-toll-station','stadium',
               'groundtrackfield','windmill','trainstation','harbor','overpass',
               'baseballfield','tenniscourt','bridge','basketballcourt','airplane',
               'ship','storagetank','Expressway-Service-area','airport','dam')
    
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]