from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from numpy import dot
from numpy.linalg import norm

threshold = 0.34

def simVoice(audio1_path,audio2_path):
    wav1 = preprocess_wav(Path(audio1_path))
    wav2 = preprocess_wav(Path(audio2_path))

    encoder = VoiceEncoder()

    embed1 = encoder.embed_utterance(wav1)
    embed2 = encoder.embed_utterance(wav2)
                            
    return dot(embed1, embed2)/(norm(embed1)*norm(embed2)) #np.inner(embed1, embed2)

ave = 0
with open("submission.csv", "w+", encoding="utf-8") as fw:
    with open("../DATA/public-test.csv", "r", encoding="utf-8") as f:
        tests = f.read().splitlines()[1:]
        fw.write("audio_1,audio_2,label" + "\n")
        for test in tests:
            audio1_path,audio2_path = test.split(",")
            audio1_path= "../DATA/public-test/" + audio1_path
            audio2_path= "../DATA/public-test/" + audio2_path

            mesSim = simVoice(audio1_path,audio2_path)
            print(mesSim)
            ave+=mesSim

            if mesSim > threshold:
                fw.write(test+",1" + "\n")
            else:
                fw.write(test+",0" + "\n")
        print(str(ave/len(tests)))

