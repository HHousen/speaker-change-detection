import matplotlib.pylab as plt
from matplotlib.pyplot import figure

from inference import Inference
from model import SSCDModel

DO_SCD = True

print("Loading model")
model = SSCDModel.load_from_checkpoint("short_scd_bigdata.ckpt")
print("Creating inference object")
inf = Inference(model, scd=DO_SCD)

print("Predicting on audio")
prediction, activations = inf("test_audio_similar.wav", return_scd_points=DO_SCD)

if DO_SCD:
    figure(figsize=(15, 3), dpi=80)
    plt.plot(activations)
    plt.title("SCD Scores per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Aggregated Score")
    plt.savefig("activations.png", bbox_inches="tight")

print("Printing prediction")
print(prediction)
