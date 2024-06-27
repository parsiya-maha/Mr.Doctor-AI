# Import from sub models
from . import BrainTumors
from . import LungCancer
from . import KidneyStone
from . import ToRecognize
from . import BreastCancer
from . import CervicalCancer
from . import LungMask



# The 5th model -> pipeline of [ToRecognize , another model]
def ToRecognizeAndPredictImage(image_path:str):
    rec = ToRecognize.ToRecognizePredictImage(image_path)

    # cancers ->  ['BrainTumor', 'KidneyStone', 'LungCancer']

    if rec == 'BrainTumors' :
        from .BrainTumors import BrainTumorsPredictImage as predict_func

    elif rec == "BreastCancer" :
        from .BreastCancer import BreastCancerPredictImage as predict_func

    elif rec == "CervicalCancer" :
        from .CervicalCancer import CervicalCancerPredictImage as predict_func

    elif rec == 'KidneyStone' :
        from .KidneyStone import KidneyStonePredictImage as predict_func

    elif rec == 'LungCancer' :
        from .LungCancer import LungCancerPredictImage as predict_func



    # Last result
    pre = predict_func(image_path)

    return f"{rec} : {pre}"