
# import os
# import pretty_midi
# from scipy.io import wavfile
# import IPython

# from keras.models import load_model
# import matplotlib.pyplot as plt
# import numpy as np
# import glob

# # n_x --- the number of notes (here we consider the problem as a multi-class with n_x classes)
# n_x = 79
# # max_midi_T_x --- the maximum number of notes we read in each midi_file
# max_midi_T_x = 1000
# # model_T_x --- the length of the sequences considered for the RNN model
# #model_T_x = 200
# model_T_x = 100
# # model_n_a --- the number of neurons of each layer of the network
# #model_n_a = 256
# model_n_a = 32
# #%%



# # n_x --- the number of notes (here we consider the problem as a multi-class with n_x classes)
# n_x = 79
# # max_midi_T_x --- the maximum number of notes we read in each midi_file
# max_midi_T_x = 1000
# # model_T_x --- the length of the sequences considered for the RNN model
# #model_T_x = 200
# model_T_x = 100
# # model_n_a --- the number of neurons of each layer of the network
# #model_n_a = 256
# model_n_a = 32
# def F_convert_midi_2_list(midi_file_l, max_midi_T_x):
#     """
#     read the notes within all midi files
#     truncate the length if > max_midi_T_x

#     Parameters
#     ----------
#     midi_file_l:
#         list of MIDI files
#     max_midi_T_x:
#         the maximum number of notes we read in a given midi_file

#     Returns
#     -------
#     X_list:
#         a list of np.array X_ohe of size (midi_T_x, n_x) which contains the one-hot-encoding representation of notes over time
#     """
#     X_list = []

#     for midi_file in midi_file_l:
#         # --- read the MIDI file
#         midi_data = pretty_midi.PrettyMIDI(midi_file)
#         note_l = [note.pitch for note in midi_data.instruments[0].notes]
#         midi_T_x = len(note_l) if len(note_l) < max_midi_T_x else max_midi_T_x
#         # --- convert to one-hot-encoding
        
#         X_ohe = np.zeros((midi_T_x, n_x))
#         for i in range(midi_T_x):
#             pitch = note_l[i]-1
#             X_ohe[i, pitch] = 1
#         X_list.append(X_ohe)

#     return X_list


# X_list = F_convert_midi_2_list(midi_file_l, max_midi_T_x)

# import numpy as np

# import numpy as np

# def F_get_max_temperature(proba_v, temperature=1):
#     """
#     Apply a temperature to the input probability, consider it as a multinomial distribution, and sample it.

#     Parameters
#     ----------
#     proba_v: np.array(n_x)
#         Input probability vector.
#     temperature: scalar float
#         Temperature parameter to apply to proba_v.
#         >1 leads to more flattened probability,
#         <1 leads to more peaky probability.

#     Returns
#     -------
#     index_pred: scalar int
#         Position of the sampled data in the probability vector.
#     pred_v: np.array(n_x)
#         Modified probability.
#     """

#     # Apply temperature to the log probabilities
#     log_proba_v_scaled = np.log(proba_v) / temperature

#     # Rescale back to probabilities
#     proba_v_scaled = np.exp(log_proba_v_scaled)

#     # Normalize to ensure the sum is 1.0
#     pred_v = proba_v_scaled / np.sum(proba_v_scaled)

#     # Ensure precision consistency by casting to float64
#     pred_v = pred_v.astype(np.float64)

#     # Explicitly cast to float64 before using in multinomial
#     pred_v = pred_v.astype(np.float64)
    
#     # Check if the sum is greater than 1.0 and normalize if needed
#     if np.sum(pred_v) > 1.0:
#         pred_v /= np.sum(pred_v)

#     # Sample from the modified probabilities
#     index_pred = np.argmax(np.random.multinomial(1, pred_v, 1))

#     return index_pred, pred_v




# def F_sample_new_sequence(model, prior_v):
#     """
#     sample the trained language model to generate new data

#     Parameters
#     ----------
#     model:
#         trained language model

#     Returns
#     -------
#     note_l: list of int
#         list of generated notes (list of their index)
#     prediction_l: list of np.array(n_x)
#         list of prediction probabilies over time t (each entry of the list is one of the y[0,t,:])
#     """


#     prediction_l = []
#     note_l=[]
#     input_m=np.zeros((1,model_T_x,n_x))
#     input_m[0,0,:]=np.random.multinomial(1,prior_v,1)[0,:]
#     for t in range(0,model_T_x-1):
#       output_m = model(input_m)
#       proba_v=output_m[0,t,:]
#       index_pred, pred_v = F_get_max_temperature(proba_v)
#       input_m[0,t+1, index_pred]=1

#       if t>-1:
#         note_l.append(index_pred)
#         prediction_l.append(pred_v)


#     return note_l, prediction_l


# model = load_model('C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/bach_modele.h5')






# print(note_l)
# plt.figure(figsize=(10, 6))
# plt.imshow(np.asarray(prediction_l).T, aspect='auto', origin='lower')
# plt.plot(note_l, 'ro')
# plt.set_cmap('gray_r')
# plt.grid(True)




# #%%
# from pydub import AudioSegment
# import numpy as np

# # Assume that new_midi_data.synthesize(fs=44100) returns a NumPy array
# import numpy as np
# from scipy.io.wavfile import write

# # Assume that new_midi_data.synthesize(fs=44100) returns a NumPy array
# audio_data_np = new_midi_data.synthesize(fs=44100)

# # Normalize the audio data to the range [-32768, 32767] for int16 format
# normalized_audio_data = (audio_data_np * 32767).astype(np.int16)

# # Export the NumPy array directly to a WAV file using scipy
# output_file_path = 'C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/audio.wav'
# write(output_file_path, 44100, normalized_audio_data)

# print(f"Audio exported successfully to: {output_file_path}")










# <!--<a href="{{ url_for('public.generate_music') }}" class="btn btn-secondary btn-lg">Jouer à MusicGen</a>-->
#       <!-- <script>
#         $(document).ready(function() {
#             $('#playMusic').click(function() {
#                 // Faites une requête AJAX pour déclencher la génération et la lecture de la musique
#                 $.ajax({
#                     url: '{{ url_for('public.generate_music') }}',
#                     type: 'POST',
#                     success: function(data) {
#                         // La requête a réussi, vous pouvez ajouter ici le code pour jouer la musique
#                         // Par exemple, en utilisant un lecteur audio HTML5
        
#                         // Convertissez la chaîne base64 en tableau de bytes
#                         var byteCharacters = atob(data.audio_data);
        
#                         // Convertissez les bytes en tableau tampon
#                         var byteNumbers = new Array(byteCharacters.length);
#                         for (var i = 0; i < byteCharacters.length; i++) {
#                             byteNumbers[i] = byteCharacters.charCodeAt(i);
#                         }
#                         var byteArray = new Uint8Array(byteNumbers);
        
#                         // Créez un objet Blob avec les données audio
#                         var blob = new Blob([byteArray], { type: 'audio/wav' });
        
#                         // Créez une URL d'objet avec le Blob
#                         var audioUrl = URL.createObjectURL(blob);
        
#                         // Créez un nouvel élément audio
#                         var audio = new Audio(audioUrl);
        
#                         // Jouez le fichier audio
#                         audio.play();
#                     },
#                     error: function() {
#                         console.error('Erreur lors de la génération de la musique');
#                     }
#                 });
#             });
#         });
#         </script> -->