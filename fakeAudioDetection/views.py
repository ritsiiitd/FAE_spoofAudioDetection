from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
import sys
import os
import joblib
import soundfile as sf
from shared.feature_extraction import get_all_features_from_sample
sp = None
fn = None
fp = None
# Create your views here.
#this is our main frontend code
def index(request):
    if request.method == 'POST' and request.POST.get('check')!='ok':
        
        Uploadedfile = request.FILES['audio']
        fs = FileSystemStorage()
        fs.save(Uploadedfile.name,Uploadedfile)

        MODEL = "models/ADC_trained_model.sav"
        FILE_PATH = 'media/'+Uploadedfile.name
        my_variable = request.POST.get('my_variable')
        check = request.POST.get('check')
        print(my_variable)
        sample_features = get_all_features_from_sample(FILE_PATH)
        model = joblib.load(MODEL)
        results = model.predict([sample_features])
        input_file = 'media/'+Uploadedfile.name
        output_file = os.path.splitext(input_file)[0]+".wav"

        # Load the .flac file
        data, sample_rate = sf.read(input_file)

        # Convert to .wav and save the file
        sf.write(output_file, data, sample_rate)
        res = "fake" if results[0] == 1 else "real"
        print("The sample audio is " + res)
        sp = sample_features
        fn = Uploadedfile.name
        fp = FILE_PATH
        return render(request,'base.html',context = {'path':output_file,'name':Uploadedfile.name,'realFile':FILE_PATH,'features':sample_features})
    
    if request.method == 'POST' and request.POST.get('check')=='ok':
        
        
        MODEL = "models/ADC_trained_model.sav"# here is our trained model
        FILE_PATH = request.POST.get('file')
        print(FILE_PATH)
        sample_features = get_all_features_from_sample(FILE_PATH)
        model = joblib.load(MODEL)
        results = model.predict([sample_features])
        res = "fake" if results[0] == 1 else "real"
        print("The sample audio is " + res)
        return render(request,'base.html',context = {'path':FILE_PATH,'res':res})
    

    return render(request,'base.html')

def play(request):
    print("insuoiide o,akplay")
    print(fn)
    print(fp)
    print(sp)
    return render(request,'play.html')