import librosa
import numpy as np
import json
import sys

def analyze_audio(file_path, output_json):
    # Load audio metadata
    y, sr = librosa.load(file_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    analysis_data = {"duration": duration, "segments": []}
    
    # Process in 10-second segments to manage RAM
    for start in range(0, int(duration), 10):
        segment = y[start*sr : (start+10)*sr]
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        rms = librosa.feature.rms(y=segment)
        
        analysis_data["segments"].append({
            "start": start,
            "mfcc": np.mean(mfcc, axis=1).tolist(),
            "energy": float(np.mean(rms))
        })
        
    with open(output_json, 'w') as f:
        json.dump(analysis_data, f)

if __name__ == "__main__":
    analyze_audio(sys.argv[1], "analysis.json")
