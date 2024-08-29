import time
from celery import Celery
from celery.utils.log import get_task_logger
from flask import Flask ,request,send_from_directory
import os
from werkzeug.utils import secure_filename
import shutil
import subprocess
import re
import requests
import sys          
import xml.etree.ElementTree as ET
from pydub import AudioSegment
from datetime import datetime , timedelta

logger = get_task_logger(__name__)

app = Celery('tasks', broker='redis://localhost:5004/0', backend='redis://localhost:5004/0')


# @app.task()
# def longtime_add(folder_for_each_video,filename_without_ext,filename):
#     isExist = os.path.exists(folder_for_each_video+"/audio")
#
#
#     os.makedirs(folder_for_each_video+"/audio")
#
#     generated_audiofile=folder_for_each_video+"/audio/"+filename_without_ext+".mp3"
#     convert_to_audio=" ffmpeg -i "+ folder_for_each_video+"/"+filename+" -vn -acodec libmp3lame -ac 1 -ab 160k -ar 16000 "+generated_audiofile
#     os.system(convert_to_audio)
#     full_wav="sox " + generated_audiofile +" "+folder_for_each_video+"/audio/"+filename_without_ext+".wav"
#     os.system(full_wav)
#
#     os.makedirs(folder_for_each_video+"/audio/"+"mp3")
#
#     audio_split="mp3splt -s -p th=-35,min=0.4,rm=50_50,trackjoin=2.5 "+generated_audiofile+" -o @f-@n -d "+folder_for_each_video+"/audio/"+"mp3"
#     os.system(audio_split)
#     audio_split_text="mp3splt -s -P -p th=-35,min=0.4,rm=50_50,trackjoin=2.5 -o _@m:@s.@h_@M:@S.@H "+generated_audiofile+" > "+folder_for_each_video+"/audio/time_o.txt"
#     os.system(audio_split_text)
#     os.makedirs(folder_for_each_video+"/audio/mp3/waves")
#
#     for files in os.listdir(folder_for_each_video+"/audio/mp3/"):
#         if files.endswith(".mp3"):
#             filenames_without_ext=os.path.splitext(files)[0]
#             wav_convert="sox "+folder_for_each_video+"/audio/mp3/"+files+" "+folder_for_each_video+"/audio/mp3/waves/"+filenames_without_ext+".wav"
#             os.system(wav_convert)
#         else:
#             continue
#
#
#     model_script="python3 check.py "+folder_for_each_video
#     os.system(model_script)
#     xml_script="python3 xml_create.py "+folder_for_each_video
#     os.system(xml_script)
#
#
#
#     st=""
#     with open(folder_for_each_video+"/transcript.xml",'r')as sendt:
#         st=sendt.read()
#     remover="rm -rf "+folder_for_each_video
#     os.system(remover)
#
#     return st
#
########################################################### Local server only English ##########################################################
# @app.task()
# def longtime_add(folder_for_each_video,filename_without_ext,filename):
#     isExist = os.path.exists(folder_for_each_video+"/audio")
#
#
#     os.makedirs(folder_for_each_video+"/audio")
#
#     generated_audiofile=folder_for_each_video+"/audio/"+filename_without_ext+".mp3"
#     convert_to_audio=" ffmpeg -i "+ folder_for_each_video+"/"+filename+" -vn -acodec libmp3lame -ac 1 -ab 160k -ar 16000 "+generated_audiofile
#     os.system(convert_to_audio)
#
#     model_script="python3 whisperX/local_server.py "+generated_audiofile +" "+folder_for_each_video
#     #print("python3 whisperX/local_server.py ",generated_audiofile ," ",folder_for_each_video)
#     os.system(model_script)
#     #xml_script="python3 create_xml.py "+fh
#
#
#     st=""
#     with open(folder_for_each_video+"/transcript.xml",'r')as sendt:
#         st=sendt.read()
#     #remover="rm -rf "+folder_for_each_video
#     #os.system(remover)
#
#     return st

############################################## All Languages #####################################################################

@app.task()
def longtime_add(folder_for_each_video,filename_without_ext,filename,source_lang):
    if source_lang == "English":
        isExist = os.path.exists(folder_for_each_video+"/audio")

        os.makedirs(folder_for_each_video+"/audio")

        source_lang = source_lang

        generated_audiofile=folder_for_each_video+"/audio/"+filename_without_ext+".mp3"
        convert_to_audio=" ffmpeg -i "+ folder_for_each_video+"/"+filename+" -vn -acodec libmp3lame -ac 1 -ab 160k -ar 16000 "+generated_audiofile
        os.system(convert_to_audio)

        generated_audiofile =folder_for_each_video + "/audio/" + filename_without_ext + ".mp3"

        wav_file = folder_for_each_video + "/audio/" + filename_without_ext + ".wav"
        convert_to_wav = "ffmpeg -i " + generated_audiofile + " " + wav_file
        os.system(convert_to_wav)

        wav_file = folder_for_each_video + "/audio/" + filename_without_ext + ".wav"

        model_script="python3 whisperX/local_server.py "+wav_file +" "+folder_for_each_video
        os.system(model_script)

    else:
        isExist = os.path.exists(folder_for_each_video + "/audio")

        os.makedirs(folder_for_each_video+"/audio")

        source_lang = source_lang

        generated_audiofile=folder_for_each_video+"/audio/"+filename_without_ext+".mp3"
        convert_to_audio=" ffmpeg -i "+ folder_for_each_video+"/"+filename+" -vn -acodec libmp3lame -ac 1 -ab 160k -ar 16000 "+generated_audiofile
        os.system(convert_to_audio)
        full_wav="sox " + generated_audiofile +" "+folder_for_each_video+"/audio/"+filename_without_ext+".wav"
        os.system(full_wav)

        os.makedirs(folder_for_each_video+"/audio/"+"mp3")

        audio_split="mp3splt -s -p min=0.4,rm=50_50,trackjoin=2.5 "+generated_audiofile+" -o @f-@n -d "+folder_for_each_video+"/audio/"+"mp3"
        os.system(audio_split)
        audio_split_text="mp3splt -s -P -p min=0.4,rm=50_50,trackjoin=2.5 -o _@m:@s.@h_@M:@S.@H "+generated_audiofile+" > "+folder_for_each_video+"/audio/time_o.txt"
        os.system(audio_split_text)
        os.makedirs(folder_for_each_video+"/audio/mp3/waves")

        for files in os.listdir(folder_for_each_video+"/audio/mp3/"):
            if files.endswith(".mp3"):
                filenames_without_ext=os.path.splitext(files)[0]
                wav_convert="sox "+folder_for_each_video+"/audio/mp3/"+files+" "+folder_for_each_video+"/audio/mp3/waves/"+filenames_without_ext+".wav"
                os.system(wav_convert)
            else:
                continue

        if source_lang == "Hindi":
            model = "ai4bharat/indicwav2vec-hindi"         

        elif source_lang == "Tamil":
            model = "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250" 
        
        elif source_lang == "Telugu":
            model = "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100"

        elif source_lang == "Kannada":
            model = "Harveenchadha/vakyansh-wav2vec2-kannada-knm-56"

        elif source_lang == "Malayalam":
            model = "Harveenchadha/vakyansh-wav2vec2-malayalam-mlm-8"
        
        elif source_lang == "Marati":
            model = "ravirajoshi/wav2vec2-large-xls-r-300m-marathi-lm-boosted"
        
        elif source_lang == "Sanskrit":
            model = "Harveenchadha/vakyansh-wav2vec2-sanskrit-sam-60"

        elif source_lang == "Bengali":
            model = "ai4bharat/indicwav2vec_v1_bengali"

        elif source_lang == "Punjabi":
            model = "Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10"
        
        elif source_lang == "Odia":
            model = "Harveenchadha/vakyansh-wav2vec2-odia-orm-100"
        
        elif source_lang == "Gujarati":
            model = "Harveenchadha/vakyansh-wav2vec2-gujarati-gnm-100"
            

        model_script="python3 indian_languages.py "+folder_for_each_video + " " + model
        os.system(model_script)
        xml_script="python3 xml_create_2.py "+folder_for_each_video + " " + source_lang
        os.system(xml_script)

        remover = "rm -rf " + folder_for_each_video + "/audio/"
        os.system(remover)

    st=""
    with open(folder_for_each_video+"/transcript.xml",'r')as sendt:
        st=sendt.read()

    return st
















##############################################################                Translate       #####################################################################################

def parseXML(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    lines = []
    for item in root.findall('line'):
        words=[]
        news = {}
        line=""
        for child in item.findall('word'):
              if child.text is not None:
                line +=child.text+" "
        lines.append(line)
    return lines

def compareXMLforTimeStamps(xmlfile,tXML):
    count=[]
    cn1=0
    cn2=0
    fu=""
    with open (xmlfile,"r",encoding='utf-8') as re:
        with open (tXML,"r",encoding='utf-8') as wr:
            re1 = re.readline()
            wr1 = wr.readline()
            while re1 !="":
                if "line timestamp" in re1:
                    cn1+=1
                    count.append(re1)
                re1 = re.readline()

            while  wr1 != "":
                if "line timestamp" in wr1:
                    cn2+=1
                    wr1=count[cn2-1]
                fu+=wr1
                wr1 = wr.readline()
    return fu

@app.task()
def getTranslation(folder_for_each_video,folder_for_each_video1,filename,source_lang,destination_lang):
    print(source_lang)
    print(destination_lang)
    sr='en'
    dt='hi'
    
    ##source lang
    if source_lang=="hindi":
        sr='hi'
    elif source_lang=='english':
        sr='en'
     
    ##destination lang   
    if destination_lang=="hindi":
        dt='hi'
    elif destination_lang=='english':
        dt='en'
        
    print(sr)
    print(dt)
    if sr==dt:
        if dt=='en':
            dt='hi'
        else:
            dt='en'
    print(sr,dt," ")
    # final_translation=[]
    initial_translation=[]
    for line in parseXML(folder_for_each_video+"/"+filename):
        # payload = {"sentence": line}
        # req = requests.post('https://udaaniitb2.aicte-india.org:8000/udaan_project_layout/translate/'+sr+'/'+dt.format("math,phy", "1"), data = payload, verify=False)
        # if(req.status_code==200):
        #     print(req.json()['translation'])
        #     final_translation.append(req.json()['translation'])
        # else :
        #     while(req.status_code!=200):
        #         req = requests.post('https://udaaniitb2.aicte-india.org:8000/udaan_project_layout/translate/'+sr+'/'+dt.format("math,phy", "1"), data = payload, verify=False)
        #     final_translation.append(req.json()['translation'])

####################### Changes Made by Akshay ###########################
        initial_translation.append(line)

    print(initial_translation)
    with open(folder_for_each_video+"/"+source_lang+'.txt','w',encoding='utf-8') as srt:
        for aa in initial_translation:
            srt.write(aa+"\n")

    #bash /home/kcdh/rishabh/Asr_Backend/indicTrans_model_files/indicTrans/joint_translate.sh /home/kcdh/rishabh/Asr_Backend/indicTrans_model_files/indicTrans/input.txt /home/kcdh/rishabh/Asr_Backend/indicTrans_model_files/indicTrans/output.txt en hi /home/kcdh/rishabh/Asr_Backend/indicTrans_model_files/en-indic/

    command = "cd ../indicTrans_model_files/indicTrans/ && " + "bash /home/kcdh/rishabh/Asr_Backend/indicTrans_model_files/indicTrans/joint_translate.sh"+" /home/kcdh/rishabh/Asr_Backend/asr_backend/" + folder_for_each_video1 +"/"+ source_lang+ ".txt" + " /home/kcdh/rishabh/Asr_Backend/asr_backend/"+ folder_for_each_video1 + "/" + destination_lang + ".txt" + " "+ sr + " " + dt + " /home/kcdh/rishabh/Asr_Backend/indicTrans_model_files/en-indic/"
    os.system(command)

    with open(folder_for_each_video+"/"+ destination_lang +'.txt','r',encoding='utf-8') as dft:
        final = dft.read()

    final_translation = final.split("\n")
    print(final_translation)

        
    tXML='<?xml version="1.0" encoding="UTF-8"?>\n'+'<transcript lang="'+destination_lang+'">\n'

    for line in final_translation[:-1]:
        tXML+='<line timestamp="" speaker="Speaker_1">\n'
        for word in line.split():
            tXML+=('<word timestamp="">')
            tXML+=(word)
            tXML+=('</word>\n')
        tXML+=('</line>\n')
    tXML+=('</transcript>\n')
    with open(folder_for_each_video+"/"+destination_lang+'.xml','w',encoding='utf-8')as tr:
        tr.write(tXML)
    tXML=compareXMLforTimeStamps(folder_for_each_video+"/"+filename,folder_for_each_video+"/"+destination_lang+'.xml')
    # remover="rm -rf "+folder_for_each_video
    # os.system(remover)
    return tXML














###########################################################               TTS          #############################################################################


def parse_XML_Time(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    times = []
    # lines.append(0)
    for item in root.findall('line'):
      timestamp_str = item.get("timestamp")
      try:
          time_obj = datetime.strptime(timestamp_str, '%H:%M:%S.%f')
      except ValueError:
          try:
              time_obj = datetime.strptime(timestamp_str, '%M:%S.%f')
          except ValueError:
              print("Invalid timestamp format")
      milliseconds = int(time_obj.microsecond/1000) + time_obj.second*1000 + time_obj.minute*60*1000 + time_obj.hour*60*60*1000
      times.append((milliseconds))
    return times


@app.task()
def getTTS(folder_for_each_video,filename,output_lang):
    print(output_lang)
    audio_folder=(folder_for_each_video+"/audio")
    output_file=folder_for_each_video+'/outputAudio.mp3'
    os.makedirs(audio_folder)
    count=1
    for line in parseXML(folder_for_each_video+"/"+filename):
        print(line)
        getSpeech="python3 get_Speech.py "+'\"'+line+'\"'+' '+'\"'+output_lang+'\"'+" "+audio_folder+"/output_"+str(count)+".wav"
        os.system(getSpeech)
        count=count+1

    start_timestamps=parse_XML_Time(folder_for_each_video+"/"+filename)
    final_audio = AudioSegment.empty()
    first_start_time = start_timestamps[0]
    if first_start_time != 0:
        initial_silence_duration = first_start_time
        initial_silence = AudioSegment.silent(duration=int(initial_silence_duration))
        final_audio += initial_silence
    for i in range(0,len(os.listdir(audio_folder))):
        audio=audio_folder+"/"+"output_"+str(i+1)+".wav"
        audio_segment = AudioSegment.empty()
        try:
            audio_segment = AudioSegment.from_file(audio)
        except FileNotFoundError:
            print(f"{audio} not found")
            continue
        end_time = start_timestamps[i] + int(timedelta(milliseconds=len(audio_segment)).seconds*1000)
        print(start_timestamps[i],end_time)
        if i < len(os.listdir(audio_folder)) - 1:
            next_start_time = start_timestamps[i+1]
            if end_time > next_start_time:
                desired_duration = next_start_time - start_timestamps[i]
                speed_factor = len(audio_segment) / desired_duration
                print("speed F ",speed_factor)
                audio_segment = audio_segment.speedup(speed_factor)
            elif end_time <= next_start_time:
                # Add silence to fill the gap
                silence_duration = (next_start_time - end_time)
                silence = AudioSegment.silent(duration=silence_duration)
                audio_segment += silence
        final_audio += audio_segment

    final_audio.export(output_file, format="mp3")
    return os.path.abspath(output_file)