import sox

#Creating a transformer
tfm = sox.Transformer()
#Trim the audio between 0 to 150 sec
tfm.trim(0, 150)
#Create an output file
tfm.build('(Full Audiobook)  This Book  Will Change Everything!.wav', '(Full Audiobook)  This Book  Will Change Everything! (Trimmed Audiobook).wav')
tfm.build('CELESTIAL WHITE NOISE .wav', 'CELESTIAL WHITE NOISE (Trimmed) .wav')
#Creating a combiner
cbn = sox.Combiner()
#Pitch shift combined audio upto 3 semitones
cbn.pitch(3.0)
#Convert the output to 8000 Hz
cbn.convert(samplerate = 192000, n_channels = 1)
#Create the output file
cbn.build(['(Full Audiobook)  This Book  Will Change Everything! (Trimmed Audiobook).wav', 'CELESTIAL WHITE NOISE (Trimmed) .wav'], 'noisy_output.wav', 'mix')
