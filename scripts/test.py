from model_predict import model_predictor
from tool_predict import tool_precitor

mp = model_predictor()
tp = tool_precitor()
mp.start("c3a1only.npy",100)
tp.start("c3a1only.npy",100)